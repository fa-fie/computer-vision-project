from architectures import *
from utils import *
from eval import *
import torch
import os


def eval_attack(model, device, attack_name, batch_size=64):
    dataset = AttackTestDataset(attack_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Evaluating model on {attack_name} attack test set")
    evaluate_on_dataset(model, dataloader, device)


def attack_first_model():
    # Load model
    device = torch.device("cpu")
    model = AlexNet()
    model.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), "model", "alexnet_weights.pth"),
            map_location=device,
            weights_only=True,
        )
    )

    # Evaluate on attacks
    eval_attack(model, device, "shadow")
    eval_attack(model, device, "occlusion")
    eval_attack(model, device, "graffiti")
    eval_attack(model, device, "noise_blur")


if __name__ == "__main__":
    attack_first_model()
