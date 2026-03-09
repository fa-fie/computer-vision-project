import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# References:
# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# NOTE: We most likely need a more complex architecture like AlexNet. I started with LeNet5 to get more familiar with how to
# set things up.


class LeNet5:
    # NOTE: The code below is mostly taken from the Quickstart, with structural adjustments.

    def __init__(self):
        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        self.model = LeNet5Model().to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        # TODO: think about which LR to use
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        self.transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        self.load_GTSRB_data()

    def load_GTSRB_data(self):
        self.test_data = datasets.GTSRB(
            root="data",
            split="test",
            download=True,
            transform=self.transform,
        )
        self.train_data = datasets.GTSRB(
            root="data",
            split="train",
            download=True,
            transform=self.transform,
        )

    def save_model(self, fname="lenet5.pth"):
        path = os.path.join("data", fname)
        torch.save(self.model.state_dict(), path)


class LeNet5Model(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: untested - also we would need more output classes
        self.seq_modules = nn.Sequential(
            nn.Conv2d(1, 1, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(1, 1, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(120),
            nn.Sigmoid(),
            nn.Linear(84),
            nn.Sigmoid(),
            nn.Linear(10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq_modules(x)
