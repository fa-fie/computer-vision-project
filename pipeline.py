import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from architectures import AlexNet
from attack_classifier import AttackClassifier


# Full E2E pipeline:
#   1. AttackClassifier  -> probability "p" that image was attacked
#   2. GaussianDenoiser  -> corrected image  (TODO: not implemented yet)
#   3. Soft blend        -> p * corrected + (1-p) * original
#   4. AlexNet           -> sign class
#
# The soft blend is the key trick: instead of a hard if/else (which has no gradient),
# we mix the two images proportionally to p, so gradients can flow through everything.
class AdversarialRobustnessPipeline(nn.Module):
    def __init__(self, sign_classifier, attack_classifier, gaussian_denoiser=None):
        super().__init__()
        self.attack_classifier = attack_classifier
        self.sign_classifier = sign_classifier
        # TODO: pass in GaussianDenoiser here once implemented.
        # It needs to be an nn.Module that takes and returns (B, 3, 224, 224) tensors.
        self.gaussian_denoiser = gaussian_denoiser

    def forward(self, x):
        # get attack probability for each image in the batch
        attack_prob = torch.sigmoid(self.attack_classifier(x))  # (B, 1)

        # apply Gaussian correction if available, otherwise pass image unchanged
        # TODO: replace with self.gaussian_denoiser(x) once implemented
        corrected = self.gaussian_denoiser(x) if self.gaussian_denoiser else x

        # soft blend: p=1 -> fully corrected, p=0 -> original unchanged
        p = attack_prob.view(-1, 1, 1, 1)  # reshape to broadcast over image dims
        blended = p * corrected + (1.0 - p) * x

        sign_logits = self.sign_classifier(blended)  # (B, 43)
        return sign_logits, attack_prob

    @torch.no_grad()
    def predict(self, x, attack_threshold=0.5):
        sign_logits, attack_prob = self.forward(x)
        # return predicted sign class and a bool for whether attack was detected
        return sign_logits.argmax(dim=1), (attack_prob.squeeze(1) >= attack_threshold)


def train_attack_classifier(classifier, train_loader, val_loader, device,
                            num_epochs=15, lr=1e-3, save_path="model/attack_classifier.pth"):
    classifier = classifier.to(device)
    criterion = nn.BCEWithLogitsLoss()  # handles sigmoid internally, numerically stable
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        # training pass
        classifier.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(classifier(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # validation pass
        classifier.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                logits = classifier(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = 100.0 * correct / total

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc))
        print(f"Epoch {epoch:02d}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")

        # save whenever we beat the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(classifier.state_dict(), save_path)
            print(f"  -> saved (val_acc={val_acc:.2f}%)")

    return history


def train_end_to_end(pipeline, train_loader, val_loader, device,
                     num_epochs=10, lr=1e-4, lambda_det=0.5,
                     save_path="model/pipeline_e2e.pth"):
    # Combined loss: sign classification + attack detection weighted by lambda_det
    # total = CrossEntropy(sign) + lambda_det * BCE(attack detected)
    pipeline = pipeline.to(device)
    sign_criterion = nn.CrossEntropyLoss()
    det_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=lr)

    best_sign_acc = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        pipeline.train()
        total_loss_sum = 0.0
        for images, sign_labels, attack_labels in train_loader:
            images = images.to(device)
            sign_labels = sign_labels.to(device)
            attack_labels = attack_labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            sign_logits, _ = pipeline(images)
            # call attack_classifier directly to get raw logits for BCEWithLogitsLoss
            # (pipeline.forward already applies sigmoid, which we don't want here)
            attack_logits = pipeline.attack_classifier(images)
            loss = sign_criterion(sign_logits, sign_labels) + lambda_det * det_criterion(attack_logits, attack_labels)
            loss.backward()
            optimizer.step()
            total_loss_sum += loss.item() * images.size(0)
        total_loss_sum /= len(train_loader.dataset)

        pipeline.eval()
        sign_correct, det_correct, total = 0, 0, 0
        with torch.no_grad():
            for images, sign_labels, attack_labels in val_loader:
                images = images.to(device)
                sign_labels = sign_labels.to(device)
                attack_labels = attack_labels.to(device).unsqueeze(1)
                sign_logits, attack_prob = pipeline(images)
                sign_correct += (sign_logits.argmax(1) == sign_labels).sum().item()
                det_correct += ((attack_prob >= 0.5).float() == attack_labels).sum().item()
                total += images.size(0)

        sign_acc = 100.0 * sign_correct / total
        det_acc = 100.0 * det_correct / total

        history.append(dict(epoch=epoch, total_loss=total_loss_sum, sign_acc=sign_acc, det_acc=det_acc))
        print(f"Epoch {epoch:02d}/{num_epochs} | loss={total_loss_sum:.4f} | sign_acc={sign_acc:.2f}% | det_acc={det_acc:.2f}%")

        if sign_acc > best_sign_acc:
            best_sign_acc = sign_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(pipeline.state_dict(), save_path)
            print(f"  -> saved (sign_acc={sign_acc:.2f}%)")

    return history
