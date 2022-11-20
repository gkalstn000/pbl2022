import torch.nn as nn

class FC_layer(nn.Module) :
    def __init__(self, size_in, size_out, keep_prob = 0.8):
        linear = nn.Linear(size_in, size_out)

        self.layer = nn.Sequential(
            linear,
            nn.BatchNorm1d(size_out),
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob)
        )

    def forward(self, x):
        x = self.layer(x)
        return x