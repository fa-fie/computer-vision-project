import matplotlib.pyplot as plt
from evaluate import *
import pandas as pd
import numpy as np
import os

plot_folder = os.path.join(os.getcwd(), "plots")
plot_colors = ["midnightblue", "royalblue", "lightblue", "darkgreen", "darkseagreen"]

def plot_over_epochs(files):
    fpaths = [os.path.join(os.getcwd(), "results", csv_fname + ".csv") for csv_fname, _ in files]
    dfs = [pd.read_csv(fpath) for fpath in fpaths]

    min_loss = max(min([df.min(axis=0)["Loss"] for df in dfs]) - 0.05, 0)
    max_loss = max([df.max(axis=0)["Loss"] for df in dfs]) + 0.05

    min_acc = max(min([df.min(axis=0)["Val Acc"] for df in dfs]) - 5, 0)
    max_acc = min(max([df.max(axis=0)["Val Acc"] for df in dfs]) + 5, 102)

    for idx, (fname, title) in enumerate(files):
        df = dfs[idx]

        loss_acc_plot(df, f"{title}\nValidation Set Accuracy Over Epochs", fname + "_acc", df["Val Acc"], "Accuracy [%]", (min_acc, max_acc), plot_colors[0])
        loss_acc_plot(df, f"{title}\nCross Entropy Loss Over Epochs", fname + "_loss", df["Loss"], "Cross Entropy Loss", (min_loss, max_loss), plot_colors[1])

def loss_acc_plot(df, title, fname_prefix, y, y_label, y_lim, color):
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_xticks(df["Epoch"] + 1, minor=True)
    ax.set_xticks(range(0, len(df) + 1, 2), minor=False)
    ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
    ax.set_ylabel(y_label)
    ax.plot(df["Epoch"] + 1, y, c=color)
    ax.set_title(title)
    fig.tight_layout()
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="lightgrey", linestyle="--", linewidth=1)
    plt.savefig(os.path.join(plot_folder, fname_prefix + "_over_epochs.png"))

def plot_test_accuracies(files, width=0.1, test_sets=[("Initial", "Initial GTSRB"), ("occlusion", "Occlusion Attack"), ("shadow", "Shadow Attack"), ("noise_blur", "Noise & Blur Attack"), ("graffiti", "Graffiti Attack")]):
    # Reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="darkslategrey", linestyle="--", linewidth=0.5)

    accuracies = [get_accuracies_of_model(fname) for fname, _ in files]
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
        ("100_initial_data", "100% Initial Data"),
        ("70_initial_30_occlusion", f"30% Occlusion Attack And 70% Initial Data"),
        ("50_initial_50_occlusion", f"50% Occlusion Attack And 50% Initial Data"),
        ("30_initial_70_occlusion", f"70% Occlusion Attack And 30% Initial Data"),
        ("100_initial_100_occlusion", f"100% Occlusion Attack And 100% Initial Data"),
    ]
    plot_over_epochs(files)
    
    files = [
        ("100_initial_data", "100% Initial"),
        ("70_initial_30_occlusion", f"30% Attack &\n70% Initial"),
        ("50_initial_50_occlusion", f"50% Attack &\n50% Initial"),
        ("30_initial_70_occlusion", f"70% Attack &\n30% Initial"),
        ("100_initial_100_occlusion", f"100% Attack &\n100% Initial"),
    ]
    plot_test_accuracies(files)