# pyoephys.ml._models.py

import torch
import torch.nn as nn


class EMGRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EMGRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class EMGClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EMGClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)

