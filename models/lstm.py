"""Baseline LSTM classifier.

Small on purpose (~10k parameters with default config). The point of v1 is the
plumbing — once the training loop, dataset, and serving path all work, swapping
in a Transformer or TFT is a 30-line change.

Outputs a single logit per sequence; sigmoid is applied at loss/inference time.
"""
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # (batch, hidden_size)
        logit = self.head(last).squeeze(-1)  # (batch,)
        return logit
