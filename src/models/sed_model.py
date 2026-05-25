"""Sound Event Detection (SED) model with attention-based temporal pooling.

Key insight over clip-level EfficientNet:
  - Standard: spectrogram → CNN → GlobalPool → single prediction
  - SED:      spectrogram → CNN backbone → per-frame features
                         → per-frame logits  (what is calling?)
                         → per-frame attention (when is it calling?)
                         → attention-weighted sum → clip prediction

The model learns to focus on frames that actually contain calls,
ignoring silence. This is fundamentally more aligned with how bioacoustic
detection works — a species is "present" in a clip if it calls at any point.

Architecture detail
-------------------
Input (B, 1, 128, 497):
  → EfficientNet-B3 backbone (no global pool) → (B, 1536, 4, 15)
  → Mean over mel axis (dim=2)                → (B, 1536, 15)   [15 time frames]
  → Permute                                   → (B, 15, 1536)
  → Linear (shared across time frames)
       frame_logits  → (B, 15, num_classes)
       frame_attn    → (B, 15, num_classes)
  → clip_probs = sum_t [ softmax(attn)_t  ×  sigmoid(logit)_t ]
  → clip_logits = logit(clip_probs)           → (B, num_classes)

Output is compatible with BCEWithLogitsLoss — same interface as EfficientNetMel.
"""

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class SEDModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert _HAS_TIMM, "Install timm: pip install timm"

        # Backbone with spatial output (no global pool)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",    # <-- keep spatial feature map
        )
        feat_dim = self.backbone.num_features   # 1536 for B3

        # Two parallel linear heads — applied per time frame (shared weights)
        self.drop = nn.Dropout(dropout)
        self.fc_logits = nn.Linear(feat_dim, num_classes)   # "what"
        self.fc_attn   = nn.Linear(feat_dim, num_classes)   # "when"

        nn.init.xavier_uniform_(self.fc_logits.weight)
        nn.init.constant_(self.fc_logits.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_attn.weight)
        nn.init.constant_(self.fc_attn.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        feat = self.backbone(x)          # (B, C, H, W) — no global pool
        feat = feat.mean(dim=2)          # (B, C, W) — collapse mel/freq axis
        feat = feat.permute(0, 2, 1)     # (B, W, C) = (B, time_frames, feat_dim)
        feat = self.drop(feat)

        frame_logits = self.fc_logits(feat)              # (B, T, num_classes)
        frame_attn   = self.fc_attn(feat)                # (B, T, num_classes)

        # Attention over time: softmax so weights sum to 1 across T
        attn_weights = torch.softmax(frame_attn, dim=1)  # (B, T, num_classes)
        frame_probs  = torch.sigmoid(frame_logits)        # (B, T, num_classes)

        # Weighted sum → clip-level probability
        clip_probs = (attn_weights * frame_probs).sum(dim=1)   # (B, num_classes)

        # Convert back to logit space so BCEWithLogitsLoss works unchanged
        clip_logits = torch.logit(clip_probs.clamp(1e-6, 1 - 1e-6))
        return clip_logits     # (B, num_classes)

    def forward_with_attention(self, x: torch.Tensor) -> tuple:
        """Returns (clip_logits, frame_probs, attn_weights) for visualization."""
        feat = self.backbone(x).mean(dim=2).permute(0, 2, 1)
        feat = self.drop(feat)
        frame_logits = self.fc_logits(feat)
        frame_attn   = self.fc_attn(feat)
        attn_weights = torch.softmax(frame_attn, dim=1)
        frame_probs  = torch.sigmoid(frame_logits)
        clip_probs   = (attn_weights * frame_probs).sum(dim=1)
        clip_logits  = torch.logit(clip_probs.clamp(1e-6, 1 - 1e-6))
        return clip_logits, frame_probs, attn_weights
