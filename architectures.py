import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=43):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # 3 input channels
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # Flattened size is 256 * 5 * 5 = 6400
        # Fact check this I dont really understand the dimensions of linear layer yet.
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



# As in the paper "Defense Against Adversarial Attacks using Convolutional Auto-Encoders" (with 3 input channels)
class LearnedDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Decoder (with space for skip connection channels)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.dec1 = nn.Conv2d(32 + 32, 3, kernel_size=3, padding=1)  # output correction

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)    # (B, 32, 224, 224)
        e2 = self.enc2(e1)   # (B, 64, 112, 112)
        e3 = self.enc3(e2)   # (B, 128, 56, 56)

        # Decode with skip connections
        d3 = self.dec3(e3)                    # (B, 64, 112, 112)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B, 128, 112, 112) -> (B, 32, 224, 224)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 64, 224, 224) -> (B, 3, 224, 224)

        # Residual: learn the correction, add to input
        return x + d1