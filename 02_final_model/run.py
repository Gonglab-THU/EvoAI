import torch
import pandas as pd
from model import PretrainModel
import sys

sys.path.append("../utils")
from loss import spearman_loss
from metrics import spearman_corr
from common import to_gpu

#######################################################################
# pretrain model
#######################################################################

dms_node_dim = 64
dms_num_layer = 1
dms_n_head = 8
dms_pair_dim = 64

#######################################################################
# predifined parameters
#######################################################################

device = 0
batch_size = 4

num_layer = 2

seed = 0
learning_rate = 1e-4
early_stop = 10

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#######################################################################
# data
#######################################################################


class BatchData(torch.utils.data.Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        mut_name = self.csv.iloc[index].name
        data_pt = torch.load(f"../mut_data/{mut_name}/ensemble.pt")["ddG"]
        for i in data_pt.keys():
            for j in data_pt[i].keys():
                data_pt[i][j] = data_pt[i][j].squeeze(0)
        return data_pt["wt"], data_pt["mut"], torch.tensor(self.csv.loc[mut_name, "score"]).to(torch.float32), mut_name


# test data
all_csv = pd.read_csv("../data.csv", index_col=0)
test_csv = all_csv.sample(frac=0.2, axis=0, random_state=seed)

# train data
train_index = [i for i in all_csv.index if i not in test_csv.index]
train_csv = all_csv.loc[train_index].copy()

train_dataset = BatchData(train_csv)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = BatchData(test_csv)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# run
model = PretrainModel(num_layer, dms_node_dim, dms_num_layer, dms_n_head, dms_pair_dim).to(device)
model.load_state_dict(torch.load("../model_fitness_Seq/model.pt", map_location="cpu").state_dict(), strict=False)

# fixed pretrain parameters
for name, param in model.named_parameters():
    if "pretrain" in name or "logits_coef" in name or "finetune" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=8, verbose=True)

best_corr = float("-inf")
stop_step = 0
loss = pd.DataFrame()
for epoch in range(50):
    # train
    model.train()
    epoch_loss = 0
    for wt, mut, label, _ in train_loader:
        wt, mut, label = to_gpu(wt, device), to_gpu(mut, device), to_gpu(label, device)
        optimizer.zero_grad()
        pred = model(wt, mut)
        train_soft_loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 0.2, "kl")
        train_soft_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
        optimizer.step()
        epoch_loss += train_soft_loss.item()
    train_loss = epoch_loss / len(train_loader)

    # test
    preds = []
    trues = []
    with torch.no_grad():
        for wt, mut, label, _ in test_loader:
            wt, mut = to_gpu(wt, device), to_gpu(mut, device)
            pred = model(wt, mut)
            preds += pred.detach().cpu().tolist()
            trues += label.detach().cpu().tolist()

    test_corr = spearman_corr(torch.tensor(preds), torch.tensor(trues))
    loss.loc[f"{epoch}", "train_loss"] = train_loss
    loss.loc[f"{epoch}", "test_corr"] = test_corr.item()
    loss.to_csv("loss.csv")

    if test_corr > best_corr:
        stop_step = 0
        best_corr = test_corr
        torch.save(model, "best.pt")
    else:
        stop_step += 1
        if stop_step >= early_stop:
            break
