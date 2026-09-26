import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


NUM_CLASSES = 5


def create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Create EfficientNet-B0 for 5-class diabetic retinopathy classification.
    """

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_b0(weights=weights)

    # EfficientNet-B0 classifier:
    # Linear(1280, 1000) for ImageNet
    #
    # Replace it with:
    # Linear(1280, 5) for DR classification.

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model