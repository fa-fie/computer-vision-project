import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# References:
# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


# NOTE: training and test loops are for now taken entirely from the above - only with slight structural adjustments.
# They are untested and need to still be adjusted.


def train(arch, batch_size=64):
    train_dataloader = DataLoader(arch.training_data, batch_size=batch_size)

    size = len(train_dataloader.dataset)
    arch.model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(arch.device), y.to(arch.device)

        # Compute prediction error
        pred = arch.model(X)
        loss = arch.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        arch.optimizer.step()
        arch.optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(arch, batch_size=64):
    test_dataloader = DataLoader(arch.testing_data, batch_size=batch_size)

    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    arch.model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in arch.dataloader:
            X, y = X.to(arch.device), y.to(arch.device)
            pred = arch.model(X)
            test_loss += arch.loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_test_for_epochs(arch, epochs=20):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(arch)
        test(arch)
    print("Done!")
