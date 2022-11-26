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

<<<<<<< HEAD
        probs = self.elu(self.fc3a(x))
        probs = self.prob_out(probs)
        probs = self.softmax(probs)
=======
        return x

    def _values(self, x):
>>>>>>> 0ea41a7f4a8e444492e6930c7f5edd7d39956e19

        val = self.elu(self.fc3b(x))
        val = self.val_out(val)

        return val

    def _probs(self, x):

        probs = self.elu(self.fc3a(x))
        probs = self.prob_out(probs)
        probs = self.softmax(probs)

        return probs

    def values(self, x):

        val = self._values(self._backbone(x))
        return val

    def prob(self, x):

        probs = self._probs(self._backbone(x))
        return probs

    def forward(self, x):

        x = self._backbone(x)
        probs = self._probs(x)
        val = self._values(x)

        return probs, val
