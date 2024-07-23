import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_pos_CB(pos14, atom_mask):

    # pos14 = [N, L, 14, 3]
    # atom_mask = [N, L, 14]

    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)

    # mask_CB = [N, L, 3]

    pos_CA = pos14[:, :, ATOM_CA, :]
    pos_CB = pos14[:, :, ATOM_CB, :]

    # pos_CA = [N, L, 3]
    # pos_CB = [N, L, 3]

    return torch.where(mask_CB, pos_CB, pos_CA)


def construct_3d_basis(center, p1, p2):
    """
    center: pos_CA
    p1: pos_C
    p2: pos_N

    Return a batch of orthogonal basis matrix: [N, L, 3, 3cols_index]
    The matrix is composed of 3 column vectors: [e1, e2, e3]
    """
    # center = [N, L, 3]
    # p1 = [N, L, 3]
    # p2 = [N, L, 3]

    v1 = p1 - center
    e1 = v1 / (torch.linalg.norm(v1, ord=2, dim=-1, keepdim=True) + 1e-6)

    v2 = p2 - center
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.linalg.norm(u2, ord=2, dim=-1, keepdim=True) + 1e-6)

    e3 = torch.cross(e1, e2, dim=-1)

    matrix = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)

    # matrix = [N, L, 3, 3_index]

    return matrix


def local_to_global(R, t, p):
    """
    R: basis matrix: [N, L, 3, 3_index]
    t: pos_CA: [N, L, 3]
    p: local coordinates: [N, L, ..., 3]
    q: global coordinates: [N, L, ..., 3]

    Convert local coordinates p to global coordinates q
    """
    assert p.shape[-1] == 3

    # R = [N, L, 3, 3_index]
    # t = [N, L, 3]
    # p = [N, L, ..., 3]

    p_size = p.shape
    N, L = p_size[:2]
    p = p.view(N, L, -1, 3).transpose(-1, -2)

    # p = [N, L, 3, ...]

    q = torch.matmul(R, p) + t.unsqueeze(-1)

    # q = [N, L, 3, ...]

    q = q.transpose(-1, -2).reshape(p_size)

    # q = [N, L, ..., 3]

    return q


def global_to_local(R, t, q):
    """
    R: basis matrix: [N, L, 3, 3_index]
    t: pos_CA: [N, L, 3]
    q: global coordinates: [N, L, ..., 3]
    p: local coordinates: [N, L, ..., 3]

    Convert global coordinates q to local coordinates p
    """
    assert q.shape[-1] == 3

    # R = [N, L, 3, 3_index]
    # t = [N, L, 3]
    # q = [N, L, ..., 3]

    q_size = q.shape
    N, L = q_size[:2]
    q = q.reshape(N, L, -1, 3).transpose(-1, -2)

    # q = [N, L, 3, ...]

    if t is None:
        p = torch.matmul(R.transpose(-1, -2), q)
    else:
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))

    # p = [N, L, 3, ...]

    p = p.transpose(-1, -2).reshape(q_size)

    # p = [N, L, ..., 3]

    return p
