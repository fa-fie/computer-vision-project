import architectures

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

data_path = 'data'
weights_path = 'model/alexnet_weights.pth'


def plot_results(images, labels, predictions):
    # GTSRB normalization values used in training
    mean = torch.tensor([0.3403, 0.3121, 0.3214]).view(3, 1, 1)
    std = torch.tensor([0.2724, 0.2608, 0.2669]).view(3, 1, 1)

    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        # Denormalize image for plotting
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        img = img.clip(0, 1)  # Ensure values are in [0, 1] range

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        color = 'green' if labels[i] == predictions[i] else 'red'
        plt.title(f"True: {labels[i].item()}\nPred: {predictions[i].item()}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('inference_results.png')  # Saves plot to file
    #plt.show()


def load_test_data(data_path):
    """Handles the transformation and dataset loading/downloading."""
    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                             std=[0.2724, 0.2608, 0.2669])
    ])

    print(f"Checking for GTSRB dataset in '{data_path}'...")
    test_set = datasets.GTSRB(
        root=data_path,
        split="test",
        download=True,
        transform=transform
    )
    return test_set


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    test_set = load_test_data(data_path)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=True)

    # Load model
    model = architectures.AlexNet(num_classes=43).to(device)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded weights from {weights_path}")
    else:
        print(f"Model weights not found: {weights_path}")
        return

    model.eval()

    # Inference
    print("\n----- Inference Results -----")
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    for i in range(len(labels)):
        print(f"Image {i + 1}: True Class: {labels[i].item()} | Predicted Class: {predicted[i].item()}")

    plot_results(images, labels, predicted)



if __name__ == "__main__":
    run_inference()