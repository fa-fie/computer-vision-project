import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_over_epochs(csv_fnames, titles):
    fpaths = [os.path.join(os.getcwd(), "model", csv_fname + ".csv") for csv_fname in csv_fnames]
    dfs = [pd.read_csv(fpath) for fpath in fpaths]

    min_loss = max(min([df.min(axis=0)["Loss"] for df in dfs]) - 0.05, 0)
    max_loss = max([df.max(axis=0)["Loss"] for df in dfs]) + 0.05

    min_acc = max(min([df.min(axis=0)["Val Acc"] for df in dfs]) - 5, 0)
    max_acc = min(max([df.max(axis=0)["Val Acc"] for df in dfs]) + 5, 100)

    for idx, title in enumerate(titles):
        df = dfs[idx]
        fig, ax_left = plt.subplots()

        ax_left.set_xlabel("Epoch")
        ax_left.set_xticks(df["Epoch"] + 1, minor=True)
        ax_left.set_xticks(range(0, len(df) + 1, 2), minor=False)
        ax_left.set_ylim(ymin=min_acc, ymax=max_acc)
        ax_left.set_ylabel("Accuracy [%]")
        ax_left.plot(df["Epoch"] + 1, df["Val Acc"], c="r", label="Val. set accuracy")

        ax_right = ax_left.twinx()
        ax_right.set_ylabel("Loss")
        ax_right.set_ylim(ymin=min_loss, ymax=max_loss)
        ax_right.plot(df["Epoch"] + 1, df["Loss"], c="b", label="Cross entropy loss")

        ax_left.set_title(title)
        fig.tight_layout()
        # TODO: nicer legend placing
        fig.legend(loc="upper right", borderaxespad=5.5)
        ax_left.grid(axis="x", color="lightgrey", linestyle="--", linewidth=1)
        plt.savefig(os.path.join(os.getcwd(), "plots", csv_fnames[idx] + "_over_epochs.png"))


def plot_accuracies(model_weight_fnames):
    return


if __name__ == "__main__":
    plot_over_epochs(["first_model_weights", "adv_training_0.7_occlusion", "adv_training_0.3_occlusion"], ["Initial model", f"Adversarial training 30% occlusion", f"Adversarial training 70% occlusion"])