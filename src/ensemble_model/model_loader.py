import torch

from src.models.EfficientNetB0.model import (
    create_efficientnet_b0,
)

from src.models.DenseNet121.model import (
    create_densenet121,
)

from src.models.ResNet18.model import (
    create_resnet18,
)

from src.models.MobileNetV2.model import (
    create_mobilenet_v2,
)


MODEL_CREATORS = {
    "EfficientNetB0": create_efficientnet_b0,
    "DenseNet121": create_densenet121,
    "ResNet18": create_resnet18,
    "MobileNetV2": create_mobilenet_v2,
}


def load_model(
    model_name,
    checkpoint_path,
    device,
):
    """
    Load a trained model from checkpoint.
    """

    if model_name not in MODEL_CREATORS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model = MODEL_CREATORS[
        model_name
    ](
        num_classes=5,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model