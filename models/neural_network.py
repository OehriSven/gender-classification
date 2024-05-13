import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_features = args.in_features
        self.num_classes = args.num_classes

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=self.num_classes)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x
