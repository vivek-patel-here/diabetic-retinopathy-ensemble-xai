import torch

from src.models.EfficientNetB0.model import create_efficientnet_b0


def main():

    model = create_efficientnet_b0(
        num_classes=5,
        pretrained=True,
    )

    print(model)

    # Dummy batch
    x = torch.randn(
        4,
        3,
        224,
        224,
    )

    with torch.no_grad():
        output = model(x)

    print("\nInput shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput:")
    print(output)


if __name__ == "__main__":
    main()