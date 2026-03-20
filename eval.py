import torch
import os
import tqdm


def evaluate_on_dataset(model, dataloader, device):
    model.eval()  # set to eval mode

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"\n===============================")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"===============================")
