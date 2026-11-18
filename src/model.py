import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBiGRUScorer(nn.Module):
    def __init__(self, input_dim=228, num_classes=472, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.bigru = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc_embed = nn.Linear(hidden_dim * 2, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, 228)
        x = x.transpose(1, 2)               # -> (B, 228, T)
        x = self.conv(x)
        x = x.transpose(1, 2)               # -> (B, T, 256)
        out, _ = self.bigru(x)              # -> (B, T, 512)
        pooled = torch.mean(out, dim=1)     # -> (B, 512)
        
        embedding = F.normalize(F.relu(self.fc_embed(pooled)), p=2, dim=1)
        logits = self.classifier(embedding)
        return logits, embedding