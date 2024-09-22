import torch.nn as nn
import torch.nn.functional as F


class SimpleLSR(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleLSR, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_vector):
        hidden = F.relu(self.fc1(input_vector))
        output = F.relu(self.fc2(hidden))
        return output
