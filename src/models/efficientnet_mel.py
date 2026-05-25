import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

# torchvision EfficientNet variants — always available on Kaggle / standard PyTorch installs
_TV_VARIANTS = {
    "efficientnet_b0": ("torchvision.models", "efficientnet_b0", 1280),
    "efficientnet_b1": ("torchvision.models", "efficientnet_b1", 1280),
    "efficientnet_b2": ("torchvision.models", "efficientnet_b2", 1408),
    "efficientnet_b3": ("torchvision.models", "efficientnet_b3", 1536),
    "efficientnet_b4": ("torchvision.models", "efficientnet_b4", 1792),
}


def _build_torchvision_efficientnet(model_name: str, in_chans: int, num_classes: int) -> nn.Module:
    """Build an EfficientNet from torchvision with adapted first conv (no timm required)."""
    import torchvision.models as tvm
    builder = getattr(tvm, model_name)
    model   = builder(weights=None)

    # Replace first conv: original is Conv2d(3, C, 3, stride=2, padding=1, bias=False)
    old_conv  = model.features[0][0]
    new_conv  = nn.Conv2d(
        in_chans, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    # Initialise by averaging pretrained RGB weights into in_chans channels
    # (harmless when pretrained=False since weights are random anyway)
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans)
    model.features[0][0] = new_conv

    # Replace classifier head
    feat_dim = _TV_VARIANTS[model_name][2]
    model.classifier[1] = nn.Linear(feat_dim, num_classes)
    return model, feat_dim


class EfficientNetMel(nn.Module):
    """EfficientNet on single-channel log-mel spectrograms.

    Uses timm when available (training), falls back to torchvision automatically
    (Kaggle offline environment where timm may not be installed).
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        n_scales: int = 1,
    ):
        super().__init__()
        in_chans   = max(n_scales, 1)
        self.n_scales  = n_scales
        self._use_timm = False

        if _HAS_TIMM:
            self._use_timm = True
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="avg",
            )
            feat_dim  = self.backbone.num_features
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            # torchvision fallback — works offline on Kaggle
            assert model_name in _TV_VARIANTS, (
                f"torchvision fallback supports: {list(_TV_VARIANTS)}. Got: {model_name}"
            )
            self.backbone, feat_dim = _build_torchvision_efficientnet(
                model_name, in_chans, num_classes
            )
            self.head = None  # head is inside torchvision model's classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_timm:
            return self.head(self.backbone(x))
        else:
            return self.backbone(x)
