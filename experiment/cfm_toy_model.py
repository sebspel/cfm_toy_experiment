import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFMToyModel(nn.Module):
    """CFM onto circle with learned pairwise interactions"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),  # input: [x, y, t]
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # output: velocity field
        )

    def forward(self, x):
        return self.net(x)
