"""Audio Spectrogram Transformer (AST) wrapper.

Pretrained checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593
  - ViT-base backbone, trained on AudioSet 527 classes
  - Hidden size: 768, 12 transformer layers at self.ast.layers[i]

Training strategy:
  - Phase 1 (warmup epochs): only last 4 encoder layers + classifier train
  - Phase 2: call model.unfreeze_all() + drop LR 10× for full fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ASTModel as HFASTModel
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        freeze_layers: int = 8,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        assert _HAS_HF, "Install transformers: pip install transformers"

        self.ast = HFASTModel.from_pretrained(CHECKPOINT)

        if gradient_checkpointing and hasattr(self.ast, "gradient_checkpointing_enable"):
            self.ast.gradient_checkpointing_enable()

        # Freeze early layers. HF transformers >= 4.40 uses self.ast.layers (not encoder.layer).
        transformer_layers = self.ast.layers   # ModuleList
        for i, layer in enumerate(transformer_layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        hidden_size = self.ast.config.hidden_size   # 768 for ViT-base
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, 0.0)

    def unfreeze_all(self):
        """Unfreeze all encoder layers for phase-2 fine-tuning."""
        for p in self.ast.parameters():
            p.requires_grad = True
        print("AST: all layers unfrozen")

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels=128, frames=497)
        # AST position embeddings were trained on (1024, 128) — 10s at 16kHz.
        # Resize time axis to 1024 so patch count matches.
        if x.shape[-1] != 1024:
            x = F.interpolate(x, size=(128, 1024), mode="bilinear", align_corners=False)
        x = x.squeeze(1).permute(0, 2, 1)    # (B, 1024, 128)
        outputs = self.ast(input_values=x)
        cls = outputs.last_hidden_state[:, 0, :]   # CLS token: (B, hidden_size)
        return self.classifier(cls)                # (B, num_classes) raw logits
