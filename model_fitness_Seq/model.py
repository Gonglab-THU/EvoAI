import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_pos_CB(pos14, atom_mask):
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA, :]
    pos_CB = pos14[:, :, ATOM_CB, :]
    return torch.where(mask_CB, pos_CB, pos_CA)


def construct_3d_basis(center, p1, p2):
    v1 = p1 - center
    e1 = v1 / (torch.linalg.norm(v1, ord=2, dim=-1, keepdim=True) + 1e-6)

    v2 = p2 - center
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.linalg.norm(u2, ord=2, dim=-1, keepdim=True) + 1e-6)

    e3 = torch.cross(e1, e2, dim=-1)

    matrix = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)
    return matrix


def local_to_global(R, t, p):
    assert p.shape[-1] == 3
    p_size = p.shape
    N, L = p_size[:2]
    p = p.view(N, L, -1, 3).transpose(-1, -2)
    q = torch.matmul(R, p) + t.unsqueeze(-1)
    q = q.transpose(-1, -2).reshape(p_size)
    return q


def global_to_local(R, t, q):
    assert q.shape[-1] == 3
    q_size = q.shape
    N, L = q_size[:2]
    q = q.reshape(N, L, -1, 3).transpose(-1, -2)
    if t is None:
        p = torch.matmul(R.transpose(-1, -2), q)
    else:
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))
    p = p.transpose(-1, -2).reshape(q_size)
    return p


class PretrainGeometricAttention(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, qk_point_dim=8, v_point_dim=8):
        super().__init__()

        self.node_dim = node_dim
        self.n_head = n_head
        self.pair_dim = pair_dim
        self.qk_point_dim = qk_point_dim
        self.v_point_dim = v_point_dim
        self.pair2head = torch.nn.Linear(pair_dim, n_head, bias=False)
        self.gamma = torch.nn.Parameter(torch.ones([1, 1, n_head], requires_grad=True))
        self.proj_q_point = torch.nn.Linear(node_dim, qk_point_dim * n_head * 3, bias=False)
        self.proj_k_point = torch.nn.Linear(node_dim, qk_point_dim * n_head * 3, bias=False)
        self.proj_v_point = torch.nn.Linear(node_dim, v_point_dim * n_head * 3, bias=False)
        self.out_transform = torch.nn.Sequential(torch.nn.Linear((n_head * pair_dim) + (n_head * v_point_dim * (3 + 3 + 1)) + (n_head * 7), node_dim * 2), torch.nn.LayerNorm(node_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim * 2, node_dim))
        self.conv2dalpha = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head * 2), torch.nn.Conv2d(n_head * 2, n_head, 3, 1, 1), torch.nn.LeakyReLU())
        self.layer_norm = torch.nn.LayerNorm(node_dim)
        self.alpha2pair = torch.nn.Sequential(torch.nn.InstanceNorm2d(n_head + pair_dim), torch.nn.Conv2d(n_head + pair_dim, pair_dim, 3, 1, 1), torch.nn.LeakyReLU())

    @staticmethod
    def _heads(x, n_head, n_ch):
        s = list(x.shape)[:-1] + [n_head, n_ch]
        return x.view(*s)

    def _pair_logits(self, z):
        logits = self.pair2head(z)
        return logits

    def _spatial_logits(self, R, t, x):
        N, L = t.shape[:2]
        query_points = self._heads(self.proj_q_point(x), self.n_head * self.qk_point_dim, 3)
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.n_head, -1)
        key_points = self._heads(self.proj_k_point(x), self.n_head * self.qk_point_dim, 3)
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.n_head, -1)
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        logits = sum_sq_dist * ((-1 * self.gamma * torch.sqrt(torch.tensor(2 / (9 * self.qk_point_dim)))) / 2)
        return logits

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2).reshape(N, L, -1)
        return feat_p2n

    def _beta_aggregation(self, alpha, R, t, p_CB):
        N, L = t.shape[:2]
        v = p_CB[:, :, None, :].expand(N, L, self.n_head, 3)
        aggr = alpha.reshape(N, L, L, self.n_head, 1) * v.unsqueeze(1)
        aggr = aggr.sum(dim=2)

        feat_points = global_to_local(R, t, aggr)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = feat_points / (torch.linalg.norm(feat_points, ord=2, dim=-1, keepdim=True) + 1e-6)
        feat_spatial = torch.cat([feat_points.reshape(N, L, -1), feat_distance, feat_direction.reshape(N, L, -1)], dim=-1)
        return feat_spatial

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L = t.shape[:2]
        value_points = self._heads(self.proj_v_point(x), self.n_head * self.v_point_dim, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.n_head, self.v_point_dim, 3))
        aggr_points = alpha.reshape(N, L, L, self.n_head, 1, 1) * value_points.unsqueeze(1)
        aggr_points = aggr_points.sum(dim=2)
        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = feat_points / (torch.linalg.norm(feat_points, ord=2, dim=-1, keepdim=True) + 1e-6)
        feat_spatial = torch.cat([feat_points.reshape(N, L, -1), feat_distance.reshape(N, L, -1), feat_direction.reshape(N, L, -1)], dim=-1)
        return feat_spatial

    def forward(self, R, t, p_CB, x, z, mask):

        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        logits_sum = self.conv2dalpha(torch.cat((logits_pair, logits_spatial), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        N, L = logits_sum.shape[:2]
        mask_row = mask.view(N, L, 1, 1).expand_as(logits_sum)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
        logits_sum = torch.where(mask_pair, logits_sum, logits_sum - 1e6)
        alpha = torch.softmax(logits_sum, dim=2)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)
        feat_beta = self._beta_aggregation(alpha, R, t, p_CB)
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_spatial, feat_beta], dim=-1))
        feat_all = torch.where(mask.unsqueeze(-1), feat_all, torch.zeros_like(feat_all))
        x = self.layer_norm(x + feat_all)
        return x, self.alpha2pair(torch.cat((z, alpha), dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class PretrainEncoder(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.fixed_embedding_transform = torch.nn.Sequential(torch.nn.Linear(node_dim + 7, node_dim), torch.nn.LeakyReLU(), torch.nn.Linear(node_dim, node_dim))
        self.pair_encoder = torch.nn.Linear(7, pair_dim)
        self.blocks = torch.nn.ModuleList([PretrainGeometricAttention(node_dim, n_head, pair_dim) for _ in range(num_layer)])

    def forward(self, fixed_embedding, dynamic_embedding, pair, pos14, atom_mask):
        x = self.esm2_transform(dynamic_embedding)
        R = construct_3d_basis(pos14[:, :, ATOM_CA, :], pos14[:, :, ATOM_C, :], pos14[:, :, ATOM_N, :])
        t = pos14[:, :, ATOM_CA, :]
        p_CB = get_pos_CB(pos14, atom_mask)

        pair = self.pair_encoder(pair)
        for block in self.blocks:
            x, pair = block(R, t, p_CB, x, pair, atom_mask[:, :, ATOM_CA])

        return x


class PretrainModel(torch.nn.Module):

    def __init__(self, node_dim, n_head, pair_dim, num_layer):
        super().__init__()

        self.pretrain_encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.pretrain_mlp = torch.nn.Linear(node_dim, 20)
        self.logits_coef = torch.nn.Parameter(torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], requires_grad=True))

    def forward(self, data):

        plddt = torch.sign(torch.relu(data["fixed_embedding"][:, :, -1] - 0.7)).bool()
        data["atom_mask"] = torch.stack((data["atom_mask"], plddt.unsqueeze(-1).repeat(1, 1, 14)), dim=0).all(dim=0)

        node_feat = self.pretrain_encoder(data["fixed_embedding"][:, :, :-1], data["dynamic_embedding"], data["pair"], data["pos14"], data["atom_mask"])

        logits_pred = self.pretrain_mlp(node_feat)
        logits_pred = self.logits_coef[None, :, None, None] * torch.cat((logits_pred.unsqueeze(1), data["logits"]), dim=1)
        logits_pred = logits_pred.mean(1)
        return logits_pred
