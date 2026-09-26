import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CLASSES = 5


def create_mobilenet_v2(
    num_classes=NUM_CLASSES,
    pretrained=True
):

    weights = (
        MobileNet_V2_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model