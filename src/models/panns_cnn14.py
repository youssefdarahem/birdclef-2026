"""CNN14 architecture from PANNs (Pretrained Audio Neural Networks).

Reference: Qiuqiang Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
for Audio Pattern Recognition", IEEE/ACM TASLP 2020.

Pretrained weights (AudioSet, mAP=0.431) are downloaded automatically on first use
to ~/.cache/panns/CNN14_mAP=0.431.pth (≈300 MB).
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PANNS_WEIGHTS_URLS = [
    "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
]
PANNS_CACHE_DIR    = Path.home() / ".cache" / "panns"
PANNS_WEIGHTS_FILE = PANNS_CACHE_DIR / "Cnn14_16k_mAP=0.438.pth"


def _download_weights() -> Path | None:
    """Download CNN14 AudioSet weights if not already cached. Returns None on failure."""
    PANNS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if PANNS_WEIGHTS_FILE.exists():
        return PANNS_WEIGHTS_FILE

    for url in PANNS_WEIGHTS_URLS:
        try:
            print(f"Downloading PANNs CNN14 weights (~300 MB) → {PANNS_WEIGHTS_FILE}")
            torch.hub.download_url_to_file(url, str(PANNS_WEIGHTS_FILE), progress=True)
            return PANNS_WEIGHTS_FILE
        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")
            if PANNS_WEIGHTS_FILE.exists():
                PANNS_WEIGHTS_FILE.unlink()   # remove partial download

    print(
        "\nCould not auto-download PANNs weights. Manual download:\n"
        "  wget -O ~/.cache/panns/Cnn14_16k_mAP=0.438.pth \\\n"
        "    'https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1'\n"
        "Then pass --panns-weights ~/.cache/panns/Cnn14_16k_mAP=0.438.pth\n"
        "\nContinuing WITHOUT pretrained weights (random init)."
    )
    return None


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, pool_size)
        return x


class PANNSCNN14(nn.Module):
    """CNN14 backbone. Accepts (B, 1, n_mels, frames) log-mel input."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        weights_path: str | None = None,
    ):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1  = nn.Linear(2048, 2048)
        self.head = nn.Linear(2048, num_classes)

        self._init_weights()

        if pretrained:
            path = Path(weights_path) if weights_path else _download_weights()
            if path is not None:
                self._load_panns_weights(path)
            # else: random init, _download_weights already printed a warning

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _load_panns_weights(self, path: Path):
        """Load AudioSet pretrained weights, skipping the 527-class output head."""
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        # Skip the original 527-class classifier (fc_audioset.*) and bn0 (different input stats)
        skip_prefixes = ("fc_audioset", "bn0")
        filtered = {
            k: v for k, v in model_state.items()
            if not any(k.startswith(p) for p in skip_prefixes)
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered)
        print(f"PANNs weights loaded from {path.name}: {n_loaded} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, frames)
        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)

        # Global average + max pooling
        x = (x.mean(dim=(-2, -1)) + x.amax(dim=(-2, -1))) / 2  # (B, 2048)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.head(x)   # (B, num_classes) raw logits
