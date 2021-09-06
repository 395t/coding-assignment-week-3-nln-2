import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, act_fn, example_size, hidden_size, num_tags):
        super().__init__()
        self.fc1 = nn.Linear(example_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_tags)
        self.dropout = nn.Dropout(0.2)
        self.act_fn = act_fn
        
        # initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.act_fn(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x