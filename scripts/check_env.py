import torch
import torchvision


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("MPS available:",
          hasattr(torch.backends, "mps")
          and torch.backends.mps.is_available())


if __name__ == "__main__":
    main()