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


class EMGClassifierCNNLSTM(nn.Module):
    """
    Simplified CNN-LSTM classifier for EMG gesture recognition.
    
    Architecture:
        1. Dense layers to expand features
        2. Reshape to sequence for LSTM processing
        3. LSTM for temporal pattern learning
        4. Dense classifier head
    
    This model is designed for pre-computed EMG features, treating
    them as a temporal sequence for LSTM processing.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes
        lstm_hidden: LSTM hidden size (default: 128)
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(self, input_dim, output_dim, lstm_hidden=128, dropout=0.3):
        super(EMGClassifierCNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden = lstm_hidden
        
        # Determine sequence parameters based on input size
        # For PCA-reduced features (e.g., 30), use smaller expansion
        if input_dim < 50:
            expand_dim = 128
            self.seq_len = 16
            self.feat_per_step = 8
        else:
            expand_dim = 512
            self.seq_len = 32
            self.feat_per_step = 16
        
        # Feature expansion
        self.fc_expand = nn.Linear(input_dim, expand_dim)
        self.bn_expand = nn.BatchNorm1d(expand_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.feat_per_step,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classifier head
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.fc1 = nn.Linear(lstm_out_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # Input: (batch, input_dim)
        batch_size = x.size(0)
        
        # Expand features
        x = self.fc_expand(x)
        x = self.bn_expand(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Reshape to sequence: (batch, seq_len, feat_per_step)
        x = x.view(batch_size, self.seq_len, self.feat_per_step)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output from sequence
        x = lstm_out[:, -1, :]  # (batch, lstm_hidden*2)
        
        # Classifier
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        return x

