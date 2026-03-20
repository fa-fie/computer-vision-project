from architectures import *
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import decode_image
import torchvision.transforms as transforms


# Reference: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class ShadowOrOcclusionDataSet(Dataset):
    def __init__(self, shadow_or_occlusion="shadow"):
        self.shadow_or_occlusion = shadow_or_occlusion
        self.img_dir = os.path.join(os.getcwd(), "data", self.shadow_or_occlusion)
        self.img_labels = pd.read_csv(
            os.path.join(os.getcwd(), "data", "gtsrb", "GT-final_test.csv"), sep=";"
        )
        self.img_labels["Filename"] = self.img_labels["Filename"].map(
            lambda n: f"{n.split(".")[0]}_{self.shadow_or_occlusion}_00.png"
        )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_info = self.img_labels.iloc[idx]

        fname = img_info["Filename"]
        label = img_info["ClassId"]
        folder = f"0{label}" if label < 10 else str(label)

        img_path = os.path.join(self.img_dir, folder, fname)
        image = decode_image(img_path)

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
        image = transform(transforms.functional.to_pil_image(image))

        return image, label
