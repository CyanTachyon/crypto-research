import torch
import torch.nn as nn


class CryptoTransformerV4(nn.Module):
    def __init__(
        self,
        num_features: int = 25,
        num_pairs: int = 15,
        num_classes: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        pair_emb_dim: int = 16,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        self.pair_embedding = nn.Embedding(num_pairs, pair_emb_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features + pair_emb_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, pair_id: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, _ = x.shape
        pair_emb = self.pair_embedding(pair_id)
        pair_emb = pair_emb.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([x, pair_emb], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :T, :]

        padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        out = x[:, -1, :]
        cls_logits = self.cls_head(out)
        reg_pred = self.reg_head(out).squeeze(-1)
        return {"cls": cls_logits, "reg": reg_pred}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
