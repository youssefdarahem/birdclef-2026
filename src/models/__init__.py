from .efficientnet_mel import EfficientNetMel
from .ast_model import ASTModel
from .panns_cnn14 import PANNSCNN14
from .temporal_rnn import TemporalRNN
from .sed_model import SEDModel

__all__ = ["EfficientNetMel", "ASTModel", "PANNSCNN14", "TemporalRNN", "SEDModel"]


def build_model(name: str, num_classes: int, pretrained: bool = True, **kwargs):
    """Factory: return model by name string.

    kwargs passed through:
        weights_path (str): for panns/cnn14, path to local CNN14 .pth file
    """
    name = name.lower()
    if name.startswith("sed"):
        backbone = name.replace("sed_", "") if "_" in name else "efficientnet_b3"
        return SEDModel(num_classes=num_classes, backbone_name=backbone, pretrained=pretrained)
    elif name.startswith("efficientnet"):
        return EfficientNetMel(num_classes=num_classes, model_name=name, pretrained=pretrained)
    elif name == "ast":
        return ASTModel(num_classes=num_classes, gradient_checkpointing=True)
    elif name in ("panns", "cnn14"):
        return PANNSCNN14(
            num_classes=num_classes,
            pretrained=pretrained,
            weights_path=kwargs.get("weights_path"),
        )
    elif name in ("temporal_rnn", "rnn"):
        return TemporalRNN(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")
