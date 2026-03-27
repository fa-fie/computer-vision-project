from architectures import *
from utils import *
from run_experiments import *
import torch
import os
import tqdm

def evaluate_on_dataset(model, dataloader, device) -> float:
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

    accuracy = 100 * test_correct / test_total
    print(f"\n===============================")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"===============================")

    return accuracy


def eval_attack(model, device, attack_name, batch_size=64) -> float:
    dataset = AttackTestDataset(attack_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Evaluating model on {attack_name} attack test set")
    return evaluate_on_dataset(model, dataloader, device)


def eval_model(weights_fname, attacks=["occlusion", "shadow", "noise_blur", "graffiti"], batch_size=64):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), "model", weights_fname + ".pth"),
            map_location=device,
            weights_only=True,
        )
    )

    # Evaluate on attacks
    attack_accuracies = {attack : eval_attack(model, device, attack, batch_size) for attack in attacks}

    # Evaluate on test set
    test_dataloader = DataLoader(load_test_data(), batch_size=batch_size, shuffle=True)
    test_accuracy = evaluate_on_dataset(model, test_dataloader, device)

    return attack_accuracies, test_accuracy


if __name__ == "__main__":
    attack_accuracies, test_accuracy = eval_model("first_model_weights")
    print(attack_accuracies)
    print("Test accuracy", test_accuracy)