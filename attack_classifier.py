import torch
import torch.nn as nn


# Small CNN to detect whether an image has been attacked (1) or is clean (0).
# Much lighter than AlexNet since it's just binary classification.
class AttackClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Three conv blocks, each halving the spatial size
        self.features = nn.Sequential(
            # 224 -> 56
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),   # stabilises training
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 56 -> 14
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Collapse to a fixed 4x4 grid regardless of small size variations
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # helps with overfitting on a binary task
            nn.Linear(256, 1),  # single logit -> use BCEWithLogitsLoss during training
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    # Apply sigmoid to get a probability instead of a raw logit
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    # Returns 0 (clean) or 1 (attacked) for each image in the batch
    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).long().squeeze(1)
