import pytest
import torch

from crypto_transformer.models.transformer import CryptoTransformer
from crypto_transformer.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding


class TestCryptoTransformer:
    def test_forward_shape(self):
        model = CryptoTransformer(num_features=17, d_model=128, nhead=4, num_layers=4, num_classes=3)
        x = torch.randn(8, 48, 17)
        out = model(x)
        assert out.shape == (8, 3)

    def test_gradient_flows(self):
        model = CryptoTransformer(num_features=17, d_model=64, nhead=4, num_layers=2, num_classes=3)
        x = torch.randn(4, 16, 17)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self):
        model = CryptoTransformer(num_features=17, d_model=128, nhead=4, num_layers=4, dim_feedforward=512)
        count = model.count_parameters()
        assert 100_000 < count < 1_000_000, f"Parameter count {count} outside expected range"

    def test_single_sample(self):
        model = CryptoTransformer(num_features=17, d_model=128, nhead=4, num_layers=4)
        x = torch.randn(1, 48, 17)
        out = model(x)
        assert out.shape == (1, 3)

    def test_variable_length(self):
        model = CryptoTransformer(num_features=17, d_model=128, nhead=4, num_layers=4)
        for seq_len in [1, 16, 48, 128]:
            x = torch.randn(2, seq_len, 17)
            out = model(x)
            assert out.shape == (2, 3)


class TestPositionalEncoding:
    def test_sinusoidal_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=128, max_len=512)
        x = torch.randn(4, 48, 128)
        out = pe(x)
        assert out.shape == x.shape

    def test_learned_output_shape(self):
        pe = LearnedPositionalEncoding(d_model=128, max_len=512)
        x = torch.randn(4, 48, 128)
        out = pe(x)
        assert out.shape == x.shape

    def test_sinusoidal_deterministic(self):
        pe = SinusoidalPositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)
