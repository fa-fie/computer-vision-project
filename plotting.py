import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_over_epochs(csv_fname):
    fpath = os.path.join(os.getcwd(), "model", csv_fname + ".csv")
    df = pd.read_csv(fpath)
    fig, ax_left = plt.subplots()

    ax_left.set_xlabel("Epoch")
    ax_left.set_xticks(df["Epoch"] + 1, minor=True)
    ax_left.set_xticks(range(0, len(df) + 1, 2), minor=False)
    ax_left.set_ylim(ymin=92, ymax=100)
    ax_left.set_ylabel("Accuracy [%]")
    ax_left.plot(df["Epoch"] + 1, df["Val Acc"], c="r", label="Val. set accuracy")

    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Loss")
    ax_right.set_ylim(ymin=0, ymax=1.3)
    ax_right.plot(df["Epoch"] + 1, df["Loss"], c="b", label="Cross entropy loss")

    ax_left.set_title(csv_fname)
    fig.tight_layout()
    # TODO: nicer legend placing
    fig.legend(loc="upper right", borderaxespad=5.5)
    ax_left.grid(axis="x", color="lightgrey", linestyle="--", linewidth=1)
    plt.savefig(os.path.join(os.getcwd(), "plots", csv_fname + "_over_epochs.png"))


def plot_accuracies(model_weight_fnames):
    return


if __name__ == "__main__":
    plot_over_epochs("first_model_weights")
    plot_over_epochs("adv_training_0.7_occlusion")