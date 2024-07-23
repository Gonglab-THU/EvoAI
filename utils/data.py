import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


class BatchData(torch.utils.data.Dataset):

    def __init__(self, csv, wt_pt, mut_pt, num_neighbor):

        self.csv = csv
        self.wt_pt = wt_pt
        self.mut_pt = mut_pt
        self.num_neighbor = num_neighbor
        amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
        self.amino_acid_dict = {}
        for index, value in enumerate(amino_acid_list):
            self.amino_acid_dict[value] = index

    def __len__(self):

        return len(self.csv)

    def __getitem__(self, index):

        mut_name = self.csv.iloc[index].name
        wt_name = mut_name.rsplit("_", 3)[0] + "_wt"

        mut_name_nopt, pH, temperature = mut_name.rsplit("_", 2)
        pH, temperature = float(pH), float(temperature) / 10
        label = torch.tensor(self.csv.iloc[index]["ddG"], dtype=torch.float32)

        wt, mut = self.wt_pt[wt_name], self.mut_pt[mut_name_nopt]

        # add fixed embedding(add logits + pH + temp)
        L = wt["pos14"].shape[0]
        wt_lpt = torch.tensor([pH, temperature])[None, :].repeat(L, 1)
        mut_lpt = torch.tensor([pH, temperature])[None, :].repeat(L, 1)

        wt["fixed_embedding"] = torch.cat((wt["1d_from_3d"][:, :7], wt_lpt, wt["1d_from_3d"][:, -1].unsqueeze(-1)), dim=-1)
        mut["fixed_embedding"] = torch.cat((mut["1d_from_3d"][:, :7], mut_lpt, mut["1d_from_3d"][:, -1].unsqueeze(-1)), dim=-1)

        # mut_pos
        wt_seq_list = [self.amino_acid_dict[i] for i in list(self.csv.loc[mut_name, "wt_seq"])]
        mut_seq_list = [self.amino_acid_dict[i] for i in list(self.csv.loc[mut_name, "mut_seq"])]
        mut_pos = torch.LongTensor(wt_seq_list) != torch.LongTensor(mut_seq_list)
        wt["mut_pos"] = mut_pos
        mut["mut_pos"] = mut_pos

        # dist limit
        coor_CA = wt["pos14"][:, ATOM_CA, :]
        mut_pos_coor_CA = coor_CA[mut_pos]
        diff = mut_pos_coor_CA.view(1, -1, 3) - coor_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)

        mask = torch.zeros([dist.shape[0]], dtype=torch.bool)
        mask[dist.min(dim=1)[0].argsort()[: self.num_neighbor]] = True

        wt = mask_dict(wt, mask)
        mut = mask_dict(mut, mask)

        return wt, mut, label, mut_name


def mask_dict(data, mask):

    out = {}
    for k, v in data.items():
        if k in ("fixed_embedding", "dynamic_embedding", "pos14", "atom_mask", "mut_pos"):
            out[k] = v[mask]
        elif k in ("pair"):
            out[k] = v[mask][:, mask]
        else:
            continue
    return out


def padding(x, max_length, value):

    length = x.shape[0]
    assert length <= max_length
    if length == max_length:
        return x
    else:

        # pair
        if x.shape == (length, length, 7):
            pad = torch.full((max_length - length, length, 7), fill_value=value).to(x)
            x = torch.cat([x, pad], dim=0)
            pad = torch.full((max_length, max_length - length, 7), fill_value=value).to(x)
            return torch.cat([x, pad], dim=1)

        else:
            pad_size = [max_length - length] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)


def get_pad_value(key):

    pad_values = {"fixed_embedding": -float("99999"), "dynamic_embedding": -float("99999"), "pos14": float("99999"), "atom_mask": False, "pair": -float("99999"), "mut_pos": False}
    return pad_values[key]


def collate_fn(batch):

    wts, muts, labels, mut_names = zip(*batch)
    max_length = max([i["fixed_embedding"].shape[0] for i in wts])

    batch_wt = {}
    batch_mut = {}
    for i in wts:
        for k, v in i.items():
            if k in ("pos14", "atom_mask", "dynamic_embedding", "pair", "fixed_embedding", "mut_pos"):
                if k not in batch_wt.keys():
                    batch_wt[k] = [padding(v, max_length, value=get_pad_value(k))]
                else:
                    batch_wt[k].append(padding(v, max_length, value=get_pad_value(k)))

    for i in muts:
        for k, v in i.items():
            if k in ("pos14", "atom_mask", "dynamic_embedding", "pair", "fixed_embedding", "mut_pos"):
                if k not in batch_mut.keys():
                    batch_mut[k] = [padding(v, max_length, value=get_pad_value(k))]
                else:
                    batch_mut[k].append(padding(v, max_length, value=get_pad_value(k)))

    for i in batch_wt.keys():
        batch_wt[i] = torch.stack(batch_wt[i], dim=0)
        batch_mut[i] = torch.stack(batch_mut[i], dim=0)

    return batch_wt, batch_mut, torch.stack(labels, dim=0), mut_names
