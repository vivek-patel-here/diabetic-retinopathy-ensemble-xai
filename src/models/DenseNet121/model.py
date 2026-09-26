import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


NUM_CLASSES = 5


def create_densenet121(num_classes=NUM_CLASSES, pretrained=True):

    weights = DenseNet121_Weights.DEFAULT if pretrained else None

    model = densenet121(weights=weights)

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model