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
        self.softmax = nn.Softmax()

        # define fc layers for value output
        self.fc3b = nn.Linear(2048, 512)
        self.val_out = nn.Linear(512, 1)

        # ELU activation function
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        # x is of shape (20,24)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))

        probs = self.elu(self.fc3a(x))
        probs = self.elu(self.prob_out(x))
        probs = self.softmax(probs)

        val = self.elu(self.fc3b(x))
        val = self.val_out(x)

        return probs, val