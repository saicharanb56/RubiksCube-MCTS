from torch import nn
import torch.nn.functional as F


class NNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # initialize fully connected layers here
        self.fc1 = nn.Linear(480, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        # define fc layers for probability outputs
        self.fc3a = nn.Linear(2048, 512)
        self.prob_out = nn.Linear(512, 12)
        self.softmax = nn.Softmax(dim=-1)

        # define fc layers for value output
        self.fc3b = nn.Linear(2048, 512)
        self.val_out = nn.Linear(512, 1)

        # ELU activation function
        self.elu = nn.ELU(alpha=1.0)

    def _backbone(self, x):

        # x is of shape (480,)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))

        return x

    def _values(self, x):

        val = self.elu(self.fc3b(x))
        val = self.val_out(val)

        return val

    def _logits(self, x):

        logits = self.elu(self.fc3a(x))
        logits = self.prob_out(logits)

        return logits

    def values(self, x):

        val = self._values(self._backbone(x))
        return val

    def logits(self, x):

        logits = self._logits(self._backbone(x))
        return logits

    def forward(self, x):

        x = self._backbone(x)
        logtits = self._logits(x)
        val = self._values(x)

        return logtits, val


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
    def __init__(self,
                 state_dim: int = 480,
                 hidden_dim: int = 5000,
                 resnet_dim: int = 1000,
                 num_resnet_blocks: int = 4,
                 policy_out_dim: int = 12,
                 value_out_dim: int = 1,
                 batch_norm: bool = True):
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
            self.batch_norm_c = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        self.resnet_blocks = nn.ModuleList([
            ResnetBlock(resnet_dim, batch_norm)
            for _ in range(num_resnet_blocks)
        ])

        # final layer
        self.fc_v = nn.Linear(resnet_dim, value_out_dim)
        self.fc_p = nn.Linear(resnet_dim, policy_out_dim)

    def backbone(self, x):

        # first layer
        x = self.fc_a(x)
        if self.batch_norm:
            x = self.batch_norm_a(x)
        x = F.relu(x)

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

        if self.batch_norm:
            x = self.batch_norm_c(x)

        return F.relu(x)

    def values(self, x):
        x = self.backbone(x)
        return self.fc_v(x)

    def policy(self, x):
        x = self.backbone(x)
        return self.fc_p(x)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc_p(x), self.fc_v(x)
