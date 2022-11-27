import torch
from torch import nn


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
