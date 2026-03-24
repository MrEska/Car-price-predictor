import numpy as np
import torch
from torch import nn


class CarPriceModel(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities):
        super().__init__()

        emb_dims = [
            min(16, max(4, int(np.ceil(np.log2(cardinality + 1)))))
            for cardinality in cat_cardinalities
        ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)
        ])

        input_dim = num_numeric + sum(emb_dims)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x_num, x_cat):
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + cat_embs, dim=1)
        out = self.mlp(x)
        return out.squeeze(1)