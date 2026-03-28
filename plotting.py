import matplotlib.pyplot as plt
from evaluate import *
import pandas as pd
import numpy as np
import os

plot_folder = os.path.join(os.getcwd(), "plots")
plot_colors = ["midnightblue", "royalblue", "lightblue", "darkgreen", "darkseagreen"]

def plot_over_epochs(files):
    fpaths = [os.path.join(os.getcwd(), "model", csv_fname + ".csv") for csv_fname, _ in files]
    dfs = [pd.read_csv(fpath) for fpath in fpaths]

    min_loss = max(min([df.min(axis=0)["Loss"] for df in dfs]) - 0.05, 0)
    max_loss = max([df.max(axis=0)["Loss"] for df in dfs]) + 0.05

    min_acc = max(min([df.min(axis=0)["Val Acc"] for df in dfs]) - 5, 0)
    max_acc = min(max([df.max(axis=0)["Val Acc"] for df in dfs]) + 5, 100)

    for idx, (fname, title) in enumerate(files):
        df = dfs[idx]
        fig, ax_left = plt.subplots()

        ax_left.set_xlabel("Epoch")
        ax_left.set_xticks(df["Epoch"] + 1, minor=True)
        ax_left.set_xticks(range(0, len(df) + 1, 2), minor=False)
        ax_left.set_ylim(ymin=min_acc, ymax=max_acc)
        ax_left.set_ylabel("Accuracy [%]")
        ax_left.plot(df["Epoch"] + 1, df["Val Acc"], c=plot_colors[0], label="Val. set accuracy")

        ax_right = ax_left.twinx()
        ax_right.set_ylabel("Loss")
        ax_right.set_ylim(ymin=min_loss, ymax=max_loss)
        ax_right.plot(df["Epoch"] + 1, df["Loss"], c=plot_colors[1], label="Cross entropy loss")

        ax_left.set_title(title)
        fig.tight_layout()
        # TODO: nicer legend placing
        fig.legend(loc="upper right", borderaxespad=5.5)
        ax_left.set_axisbelow(True)
        ax_left.grid(axis="x", color="lightgrey", linestyle="--", linewidth=1)
        plt.savefig(os.path.join(plot_folder, fname + "_over_epochs.png"))


def plot_test_accuracies(files, width=0.1, test_sets=[("Initial", "Initial GTSRB"), ("occlusion", "Occlusion Attack"), ("shadow", "Shadow Attack"), ("noise_blur", "Noise & Blur Attack"), ("graffiti", "Graffiti Attack")]):
    # Reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="darkslategrey", linestyle="--", linewidth=0.5)

    accuracies = [eval_model(fname) for fname, _ in files]
    for group_idx, acc_model in enumerate(accuracies):
        for bar_idx, (test_set, _) in enumerate(test_sets):
            ax.bar(group_idx + bar_idx * width, acc_model[test_set], width, lw=0.5, edgecolor="darkslategrey", color=plot_colors[bar_idx])

    ax.set_ylim(ymin=70, ymax=100)
    ax.set_ylabel("Test Set Accuracy [%]")
    ax.set_title("Test Set Accuracies for Initial Dataset and Attacks", pad=45)

    x = np.arange(len(files))
    ticks = [title for _, title in files]
    ax.set_xticks(x + width * ((len(test_sets) - 1) / 2), ticks)

    # Reference: https://matplotlib.org/stable/users/explain/axes/legend_guide.html
    labels = [label for _, label in test_sets]
    ax.legend(labels, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower left", ncols=3, mode="expand", borderaxespad=0.)

    fig.tight_layout()
    plt.savefig(os.path.join(plot_folder, "test_accuracies.png"))


if __name__ == "__main__":
    files = [
        ("first_model_weights", "100% Initial"),
        ("adv_training_0.7_occlusion", f"30% Attack &\n70% Initial"),
        ("adv_training_0.3_occlusion", f"70% Attack &\n30% Initial"),
        ("adv_training_0_occlusion", f"100% Attack &\n100% Initial"),
    ]

    plot_over_epochs(files)
    plot_test_accuracies(files)