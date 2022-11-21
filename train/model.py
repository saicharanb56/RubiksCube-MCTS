import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, resnet_dim: int, batch_norm: bool):

        super(ResnetBlock, self).__init__()

        self.resnet_dim = resnet_dim
        self.batch_norm = batch_norm

        self.fc_a = nn.Linear(in_features=resnet_dim, out_features=resnet_dim)

        self.fc_b = nn.Linear(in_features=resnet_dim, out_features=resnet_dim)

        if batch_norm:
            self.batch_norm_a = nn.BatchNorm1d(num_features=self.resnet_dim)
            self.batch_norm_b = nn.BatchNorm1d(num_features=self.resnet_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        skip = x

        x = self.fc_a(x)
        if self.batch_norm:
            x = self.batch_norm_a(x)
        x = self.relu(x)

        x = self.fc_b(x)
        if self.batch_norm:
            x = self.batch_norm_b(x)

        return x + skip


class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, resnet_dim: int,
                 num_resnet_blocks: int, out_dim: int, batch_norm: bool):
        super().__init__()
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # input layers
        self.fc_a = nn.Linear(self.state_dim, hidden_dim)
        self.fc_b = nn.Linear(hidden_dim, resnet_dim)

        # batch_norms
        if self.batch_norm:
            self.batch_norm_a = nn.BatchNorm1d(hidden_dim)
            self.batch_norm_b = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(resnet_dim, batch_norm)
            for _ in range(num_resnet_blocks)
        ])

        # final layer
        self.fc_c = nn.Linear(resnet_dim, out_dim)

        # activation fn
        self.relu = nn.ReLU()

    def forward(self, x):

        # first layer
        x = self.fc_a(x)
        if self.batch_norm:
            x = self.batch_norm_a(x)
        x = self.relu(x)

        # first layer
        x = self.fc_b(x)
        if self.batch_norm:
            x = self.batch_norm_b(x)
        x = F.relu(x)

        # resnet blocks

        skip = x

        for i, block in enumerate(self.resnet_blocks):
            if i == 0:
                x = F.relu(block(x))
            else:
                x = F.relu(block(x) + skip)

        # final layer
        x = self.fc_c(x)

        return x
