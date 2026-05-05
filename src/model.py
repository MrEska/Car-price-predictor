import numpy as np
import torch
from torch import nn


class CarPriceModel(nn.Module):
    def __init__(
        self,
        num_numeric,
        cat_cardinalities,
        hidden_dims=(256, 128, 64),
        dropout=0.2,
    ):
        super().__init__()

        # Dobieramy rozmiar embeddingu dla każdej kolumny kategorycznej.
        emb_dims = [
            min(16, max(4, int(np.ceil(np.log2(cardinality + 1)))))
            for cardinality in cat_cardinalities
        ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)
        ])

        input_dim = num_numeric + sum(emb_dims)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        # Każdą kolumnę kategoryczną przepuszczamy przez jej embedding.
        cat_embs = [
            emb(x_cat[:, i])
            for i, emb in enumerate(self.embeddings)
        ]

        # Łączymy cechy numeryczne i embeddingi w jeden wektor.
        x = torch.cat([x_num] + cat_embs, dim=1)

        # Przepuszczamy przez MLP.
        out = self.mlp(x)

        # Z [batch_size, 1] robimy [batch_size].
        return out.squeeze(1)