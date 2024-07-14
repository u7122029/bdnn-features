#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import get_results_path

import torch

from models.configs import DatasetType, LabelType, ModelType
import bokeh.plotting as plt

KEY_TO_TITLE = {
    "val_forward_accs": "Forward Val Accuracy vs. Epoch",
    "val_forward_f1s": "Forward Val F1 vs. Epoch",
    "val_forward_losses": "Forward Val Loss vs. Epoch",
    "val_backward_losses": "Backward Val Loss vs. Epoch",
}


def get_flip_freq(prefix):
    number = ""
    for x in prefix:
        if x.isdigit():
            number += x
    number = int((int(number) / 10) * 2) if number else 255
    if prefix.endswith("DNN"):
        out = [number/255, 0, 0]
    elif prefix.endswith("CNN"):
        out = [0, number/255, 0]
    else:
        out = [0, 0, number/255]
    return out


def print_best_accs(dset_version):
    """
    Prints the best accuracies for each model over a given dataset label_version.
    :param dset_version: The label_version of the dataset.
    :return: None.
    """
    for prefix in PREFIXES:
        path = Path("results") / dset_version / (prefix + ".pt")
        data = dict(np.load(str(path)))
        print(f"{dset_version} {prefix}")
        print(f"Best Val Acc: {data['best_val_acc']}")
        print(f"Best Val F1: {data['best_val_f1']}")
        print(f"Test Acc: {data['test_acc']}")
        print(f"Test F1: {data['test_f1']}")
        print()


def show_graph(dset_version: DatasetType,
               label_type: LabelType,
               key: str,
               title: str,
               y_label: str):
    """
    Displays the graph of the model's loss, accuracy or f1 (key) based on the dataset label_version.
    Also allows specification of the models you want to view.
    :param dset_version: The label_version of the dataset
    :param label_type:
    :param key:
    :return: None.
    """
    colours = ["#0c163b", "#d843d3", "#6143d8", "#4363d8", "#43d8d1", "#43d855"]
    model_type_groups = [[ModelType.DNN, ModelType.BDNN],
                         [ModelType.DNN_PCA, ModelType.BDNN_PCA],
                         [ModelType.LENET5, ModelType.BLENET5],
                         [ModelType.RESNET56, ModelType.BRESNET56]]
    model_type_groups = [[ModelType.DNN, ModelType.BDNN]]
    flip_freqs = [1, 5]
    f1 = plt.figure(title=title,
                    x_axis_label="Epoch",
                    y_axis_label=y_label,
                    y_range=(0, 1),
                    x_range=(0, 11),
                    width=800,
                    height=600)
    for original_type, bd_type in model_type_groups:
        filepath = get_results_path(dset_version, label_type, original_type, 0)
        data = torch.load(filepath)
        is_forward_epochs = torch.Tensor(data["is_forward_epochs"]).bool()
        val_forward_f1s = torch.Tensor(data[key])
        epoch_nos = torch.arange(1, len(is_forward_epochs) + 1)
        forward_epoch_nos = epoch_nos[is_forward_epochs]
        f1.line(forward_epoch_nos, val_forward_f1s, color=colours[0], legend_label="DNN")
        f1.scatter(forward_epoch_nos, val_forward_f1s, color=colours[0], size=7)

        for i, flip_freq in enumerate(flip_freqs, start=1):
            filepath = get_results_path(dset_version, label_type, bd_type, flip_freq)
            data = torch.load(filepath)
            is_forward_epochs = torch.Tensor(data["is_forward_epochs"]).bool()
            val_forward_f1s = torch.Tensor(data[key])
            epoch_nos = torch.arange(1, len(is_forward_epochs) + 1)
            forward_epoch_nos = epoch_nos[is_forward_epochs]
            f1.line(forward_epoch_nos, val_forward_f1s, color=colours[i], legend_label=f"BDNN_F{flip_freq}")
            f1.scatter(forward_epoch_nos, val_forward_f1s, color=colours[i], size=7)
    f1.legend.location = "bottom_right"
    plt.show(f1)
    """
    for prefix in prefixes:
        print(prefix)
        path = Path("results") / str(dset_version) / (prefix + ".pt")
        data = dict(np.load(str(path)))[key]
        if prefix.startswith("_"):
            print("Here")
            prefix = prefix[1:]
        #if len(data) == 0: continue

        lstyle = "solid" if "CNN" in prefix else "dashed"
        plt.plot(data[:,1], data[:,0], label=prefix, linewidth=1.2, linestyle=lstyle)

    if key.endswith("losses"):
        plt.ylim([0,6])
    plt.title(f"{KEY_TO_TITLE[key]} ({DSET_VER_TO_TITLE[dset_version]})")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.show()"""


if __name__ == "__main__":
    show_graph(DatasetType.IMAGES,
               LabelType.ALCOHOLIC,
               "val_forward_losses")
    #print_best_accs("images_alcoholic")
    #show_graph("images_alcoholic", "val_forward_f1s")
    #f = np.load("results/images_alcoholic/_CNN.npz")
    #print(float(dict(f)["test_f1"]))
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100_50"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100_50"])