import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 5


def create_resnet18(num_classes=NUM_CLASSES, pretrained=True):

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model