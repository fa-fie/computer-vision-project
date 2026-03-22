from architectures import *
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


# Reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class AttackTestDataset(Dataset):
    def __init__(self, attack_name="shadow"):
        self.attack_name = attack_name
        self.img_dir = os.path.join(os.getcwd(), "physical_adv_attack", "generated", "test", self.attack_name)
        self.img_labels = pd.read_csv(
            os.path.join(os.getcwd(), "data", "gtsrb", "GT-final_test.csv"), sep=";"
        )
        self.img_labels["Filename"] = self.img_labels["Filename"].map(
            lambda n: f"{n.split(".")[0]}_{self.attack_name}_00.png"
        )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_info = self.img_labels.iloc[idx]

        fname = img_info["Filename"]
        label = img_info["ClassId"]

        img_path = os.path.join(self.img_dir, f"{label:02d}", fname)
        image = Image.open(img_path)

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (250, 250)
                ),  # For pretrained AlexNet we need at least 224x224
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]
                ),  # Normalize according to: https://github.com/tomlawrenceuk/GTSRB-Dataloader
            ]
        )
        image = transform(image)

        return image, label
