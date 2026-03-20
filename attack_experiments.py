from architectures import *
from utils import *
from eval import *
import torch
import os


def eval_shadow_or_occlusion_attack(model, device, shadow_or_occlusion, batch_size=64):
    dataset = ShadowOrOcclusionDataSet(shadow_or_occlusion)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Evaluating model on {shadow_or_occlusion} attack test set")
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
    eval_shadow_or_occlusion_attack(model, device, "shadow")
    eval_shadow_or_occlusion_attack(model, device, "occlusion")


if __name__ == "__main__":
    attack_first_model()
