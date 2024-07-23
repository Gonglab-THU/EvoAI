import torch


class Encoder(torch.nn.Module):

    def __init__(self, num_layer):
        super().__init__()
        if num_layer == 2:
            self.esm2_transform = torch.nn.Sequential(torch.nn.LayerNorm(1280), torch.nn.Linear(1280, 640), torch.nn.LeakyReLU(), torch.nn.Linear(640, 320), torch.nn.LeakyReLU(), torch.nn.Linear(320, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 1))

    def forward(self, dynamic_embedding, mut_pos):
        x = self.esm2_transform(dynamic_embedding)
        x = x.squeeze(-1) * mut_pos
        return x.sum(1)


class PretrainModel(torch.nn.Module):

    def __init__(self, num_layer):
        super().__init__()

        self.encoder = Encoder(num_layer)
        self.finetune_rmse_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, wt, mut):

        wt_dG = self.encoder(wt["dynamic_embedding"], wt["mut_pos"])
        mut_dG = self.encoder(mut["dynamic_embedding"], mut["mut_pos"])

        return (mut_dG - wt_dG) * self.finetune_rmse_coef
