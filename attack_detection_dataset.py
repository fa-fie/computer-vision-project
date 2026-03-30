import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Standard transform used for validation and the sign classifier
GTSRB_TRANSFORM = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
])

# Augmented transform for training the attack classifier.
# Small flips/colour jitter help the model generalise without distorting
# the attack artefacts it needs to detect.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]),
])


def _resolve(path, base):
    """Return absolute path: if path is already absolute leave it, otherwise join with base."""
    p = str(path)
    return p if os.path.isabs(p) else os.path.join(base, p)


# Dataset for training the attack classifier.
# Reads manifest.csv (produced by the generator) and builds a binary dataset:
#   original image  -> label 0 (clean)
#   attacked image  -> label 1 (attacked)
# One clean + one attacked per manifest row, so it's balanced by default.
#
# The manifest stores relative paths:
#   original_path is relative to dataset_root (where your GTSRB images live)
#   output_path   is relative to the manifest's own folder (the generator output)
class AttackDetectionDataset(Dataset):
    def __init__(self, manifest_path, attack_name="occlusion", split="train",
                 dataset_root=None, transform=None):
        # use augmented transform for training, plain one for validation
        default = TRAIN_TRANSFORM if split == "train" else GTSRB_TRANSFORM
        self.transform = transform or default

        manifest_path = os.path.abspath(manifest_path)
        output_root = os.path.dirname(manifest_path)  # generator output folder

        df = pd.read_csv(manifest_path)
        # keep only rows for the chosen attack type and split
        df = df[(df["split"] == split) & (df["attack"] == attack_name)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No data found in manifest for attack='{attack_name}', split='{split}'. "
                f"Run the generator first: python physical_adv_attack/run.py"
            )

        # resolve dataset_root: use provided value, or try to infer from first original_path
        if dataset_root is None:
            # check if the path in the manifest is already absolute
            first = str(df.iloc[0]["original_path"])
            if not os.path.isabs(first):
                raise ValueError(
                    "original_path in manifest is relative but dataset_root was not provided. "
                    "Pass dataset_root='/path/to/your/gtsrb/archive' to the dataset."
                )
        self.dataset_root = dataset_root
        self.output_root = output_root

        # build a flat list of (image_path, label) pairs
        self.samples = []
        for _, row in df.iterrows():
            orig = _resolve(row["original_path"], self.dataset_root)
            attacked = _resolve(row["output_path"], self.output_root)
            self.samples.append((orig, 0))      # clean
            self.samples.append((attacked, 1))  # attacked

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # label is float because BCEWithLogitsLoss expects float targets
        return image, torch.tensor(label, dtype=torch.float32)


# Same as AttackDetectionDataset but also returns the traffic sign class.
# Used for E2E training where we need both losses: sign classification + attack detection.
class E2EDataset(Dataset):
    def __init__(self, manifest_path, attack_name="occlusion", split="train",
                 dataset_root=None, transform=None):
        self.transform = transform or GTSRB_TRANSFORM

        manifest_path = os.path.abspath(manifest_path)
        output_root = os.path.dirname(manifest_path)

        df = pd.read_csv(manifest_path)
        df = df[(df["split"] == split) & (df["attack"] == attack_name)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No data found in manifest for attack='{attack_name}', split='{split}'."
            )

        if dataset_root is None:
            first = str(df.iloc[0]["original_path"])
            if not os.path.isabs(first):
                raise ValueError(
                    "original_path in manifest is relative but dataset_root was not provided."
                )

        # each row gives us a clean and an attacked version of the same sign
        self.samples = []
        for _, row in df.iterrows():
            sign_class = int(row["class_id"])
            orig = _resolve(row["original_path"], dataset_root)
            attacked = _resolve(row["output_path"], output_root)
            self.samples.append((orig, sign_class, 0))      # clean
            self.samples.append((attacked, sign_class, 1))  # attacked

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sign_class, attack_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, sign_class, torch.tensor(attack_label, dtype=torch.float32)
