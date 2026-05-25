"""CNN + GRU temporal model.

Splits each 5s spectrogram into N sub-frames, extracts features per sub-frame
with a shared EfficientNet-B0, then runs a GRU over the temporal sequence.
"""

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

NUM_SUBFRAMES = 10          # 10 × 0.5s = 5s


class TemporalRNN(nn.Module):
    """EfficientNet-B0 feature extractor → GRU → linear classifier.

    Input: (B, 1, n_mels, frames) where frames corresponds to 5s audio.
    The spectrogram is split along the time axis into NUM_SUBFRAMES equal chunks.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        feat_dim: int = 256,
        gru_hidden: int = 512,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert _HAS_TIMM, "Install timm: pip install timm"

        backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        raw_dim = backbone.num_features  # 1280 for B0
        self.backbone = backbone
        self.proj = nn.Linear(raw_dim, feat_dim)

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, frames)
        B, C, M, T = x.shape
        subframe_len = T // NUM_SUBFRAMES

        # Extract features for each sub-frame
        frame_feats = []
        for i in range(NUM_SUBFRAMES):
            chunk = x[:, :, :, i * subframe_len : (i + 1) * subframe_len]  # (B, 1, M, subframe_len)
            feat = self.backbone(chunk)                                       # (B, raw_dim)
            feat = self.proj(feat)                                            # (B, feat_dim)
            frame_feats.append(feat)

        seq = torch.stack(frame_feats, dim=1)   # (B, NUM_SUBFRAMES, feat_dim)
        out, _ = self.gru(seq)                   # (B, NUM_SUBFRAMES, gru_hidden)
        last = out[:, -1, :]                     # take final hidden state
        return self.head(last)                   # (B, num_classes)
