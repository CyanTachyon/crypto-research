import torch
import torch.nn as nn

from crypto_transformer.models.positional import SinusoidalPositionalEncoding


class CryptoTransformer(nn.Module):
    def __init__(
        self,
        num_features: int = 17,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        num_classes: int = 3,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)

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

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        src_key_padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        out = x[:, -1, :]
        logits = self.classifier(out)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
