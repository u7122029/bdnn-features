from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torch import nn

from data import generate_dataset
from main import test
from models import get_model
from utils import get_results_path, DATA_ROOT, DEVICE, process_batch_forwards, ALL_MODEL_PAIRS, FIGURES_ROOT, \
    ALL_FLIP_FREQS
from pacmap import PaCMAP
import torch
import pandas as pd

from models.configs import DatasetType, LabelType, ModelType, ForwardConfig
import bokeh.plotting as plotting
from bokeh.palettes import Paired6, Colorblind6, Reds8, Blues8


class GraphType(Enum):
    FEATURES = 0,
    LOSSES = 1


class PlotConfig:
    def __init__(self, key, dset_version: DatasetType, label_type: LabelType, model_type: ModelType):
        self.key = key
        self.dset_version = dset_version
        self.label_type = label_type
        self.model_type = model_type

        self.legend_loc = {
            "val_forward_accs": "bottom_right",
            "val_forward_f1s": "bottom_right",
            "val_forward_losses": "top_right",
            "val_backward_losses": "bottom_right",
        }[self.key]

    def key_epoch_details(self, flip_freq):
        flip_freq_map = {
            0: 0,
            1: 1,
            10: 2,
            50: 3,
            100: 4
        }

        key_to_title = {
            "val_forward_accs": "Forward Val Accuracy vs. Epoch",
            "val_forward_f1s": "Forward Val F1 vs. Epoch",
            "val_forward_losses": "Forward Val Loss vs. Epoch",
            "val_backward_losses": "Backward Val Loss vs. Epoch",
        }

        key_to_y_label = {
            "val_forward_accs": "Accuracy",
            "val_forward_f1s": "F1 Score",
            "val_forward_losses": "Loss",
            "val_backward_losses": "Loss",
        }

        figure_kwargs = {
            "title": key_to_title[self.key],
            "y_axis_label": key_to_y_label[self.key],
        }
        if self.key.endswith("f1") or self.key.endswith("accs"):
            figure_kwargs["y_top"] = 1

        if self.key == "val_forward_losses":
            figure_kwargs["y_range"] = (0.2, 0.7)

        colour = Colorblind6[flip_freq_map[0 if not self.model_type.is_bidirectional() else flip_freq]]
        suffix = f"_{flip_freq}" if self.model_type.is_bidirectional() else ""
        line_kwargs = {
            "legend_label": f"{self.model_type.name}{suffix}",
            "color": colour
        }

        scatter_kwargs = {
            "color": colour
        }
        return figure_kwargs, line_kwargs, scatter_kwargs


def plot_line_bokeh(fig: plt.figure, pc: PlotConfig, flip_freq):
    filepath = get_results_path(pc.dset_version, pc.label_type, pc.model_type, flip_freq)
    data = torch.load(filepath)

    is_forward_epochs = torch.Tensor(data["is_forward_epochs"]).bool()
    val_forward_f1s = torch.Tensor(data[pc.key])
    epoch_nos = torch.arange(1, len(is_forward_epochs) + 1)
    forward_epoch_nos = epoch_nos[is_forward_epochs]

    _, line_kwargs, scatter_kwargs = pc.key_epoch_details(flip_freq)
    fig.line(forward_epoch_nos, val_forward_f1s, width=3, **line_kwargs)
    fig.scatter(forward_epoch_nos, val_forward_f1s, size=5, **scatter_kwargs)


def plot_line_mpl(pc: PlotConfig, flip_freq: Optional[int]):
    filepath = get_results_path(pc.dset_version, pc.label_type, pc.model_type, flip_freq)
    data = torch.load(filepath)

    is_forward_epochs = torch.Tensor(data["is_forward_epochs"]).bool()
    key_data = torch.Tensor(data[pc.key])
    print((key_data))
    epoch_nos = torch.arange(1, len(is_forward_epochs) + 1)
    forward_epoch_nos = epoch_nos[is_forward_epochs]

    _, line_kwargs, _ = pc.key_epoch_details(flip_freq)
    plt.plot(forward_epoch_nos, key_data, color=line_kwargs["color"], label=line_kwargs["legend_label"])
    #plt.savefig(f"{pc.key}_{pc.label_type}_{pc.model_type}.svg", format="svg")
    #plt.scatter(forward_epoch_nos, val_forward_f1s, color=scatter_kwargs["color"])


def show_graph(key: str,
               label_type: LabelType,
               model_type1: ModelType,
               model_type2: ModelType,
               manual_y_top: Optional[float] = None,
               manual_y_bottom: Optional[float] = None,
               manual_x_left: Optional[float] = None,
               manual_x_right: Optional[float] = None):
    pc = PlotConfig(key, DatasetType.IMAGES, label_type, model_type1)
    pcb = PlotConfig(key, DatasetType.IMAGES, label_type, model_type2)

    flip_freqs = [1, 10, 50, 100]
    figure_kwargs, line_kwargs, scatter_kwargs = pc.key_epoch_details(0)
    """f1 = plt.figure(x_axis_label="Epoch",
                    x_range=(0, 452),
                    y_range=(0.3, 1),
                    width=1100,
                    height=800,
                    **figure_kwargs)"""
    plt.figure()
    plt.grid()
    plt.title(figure_kwargs["title"])
    plt.ylabel(figure_kwargs["y_axis_label"])
    plt.xlabel("Epoch")

    #plot_line_bokeh(f1, pc, 0)
    plot_line_mpl(pc, 0)

    for flip_freq in flip_freqs:
        #plot_line_bokeh(f1, pcb, flip_freq)
        plot_line_mpl(pcb, flip_freq)

    #f1.legend.location = pc.legend_loc
    if "y_top" in figure_kwargs:
        plt.ylim(top=figure_kwargs["y_top"])
    if manual_y_top is not None:
        plt.ylim(top=manual_y_top)
    if manual_y_bottom is not None:
        plt.ylim(bottom=manual_y_bottom)

    if manual_x_left is not None:
        plt.xlim(left=manual_x_left)
    if manual_x_right is not None:
        plt.xlim(right=manual_x_right)

    plt.legend(loc="best")
    plt.savefig(f"{FIGURES_ROOT}/{pc.key}_{pc.label_type}_{pc.model_type}.svg", format="svg")


def get_features(model, dl, out_classes):
    features = []
    all_labels = []
    all_correct = []
    with torch.no_grad():
        for batch, labels in dl:
            batch = batch.to(DEVICE)
            labels = 2 * labels // out_classes
            batch = process_batch_forwards(batch, model)

            features.append(model(batch, config=ForwardConfig.FEATURES_ONLY).cpu())
            outs = 2 * model(batch, config=ForwardConfig.FORWARD).cpu().argmax(dim=1) // out_classes

            correct: torch.Tensor = outs == labels
            all_correct.append(correct)
            all_labels.append(labels)

    return features, all_labels, all_correct


def plot_features(dset_type: DatasetType, label_type: LabelType, model_type: ModelType, flip_freq: int):
    filepath = get_results_path(dset_type, label_type, model_type, flip_freq)
    state_dict = torch.load(filepath)["state_dict"]
    #colours = [Paired6[5], Paired6[0], Paired6[1], Paired6[4]]
    colours = [Blues8[0], Reds8[-3], Reds8[0], Blues8[-3]]

    _, _, dl_test = generate_dataset(DATA_ROOT, dset_type, label_type, model_type)

    model = get_model(label_type, model_type).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    out_classes = label_type.out_classes()

    test_features, test_labels, test_correct = get_features(model, dl_test, out_classes)
    test_features = torch.cat(test_features, dim=0).flatten(1)
    test_labels = torch.cat(test_labels).long()
    test_correct = torch.cat(test_correct).long()
    test_colours_idxs = 2 * test_labels + test_correct

    pacmap = PaCMAP()
    test_colours = [colours[x.item()] for x in test_colours_idxs]
    test_features = torch.Tensor(pacmap.fit_transform(test_features.numpy()))

    df = {
        "test_features_x": test_features[:, 0],
        "test_features_y": test_features[:, 1],
        "test_labels": test_labels,
        "test_correct": test_correct,
        "test_colours_idxs": test_colours_idxs,
        "test_colours": test_colours
    }
    df = pd.DataFrame(df)
    #source = plotting.ColumnDataSource.from_df(df)

    #f1 = plotting.figure()
    #f1.scatter("test_features_x", "test_features_y", source=source, color="test_colours")
    #plotting.show(f1)
    plt.figure()
    #plt.grid()
    suffix = f"_{flip_freq}" if model_type.is_bidirectional() else ""
    plt.title(f"PaCMAP Input Space Reduction ({model_type.name}{suffix} colouring)")
    for i in [1, 3, 0, 2]:
        partition = df[df["test_colours_idxs"] == i]
        plt.scatter(partition["test_features_x"], partition["test_features_y"], color=colours[i], s=1.5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f"{FIGURES_ROOT}/pacmap_{label_type}_{model_type}.svg", format="svg")


def main(graph_to_show: GraphType):
    torch.manual_seed(0)
    if graph_to_show == GraphType.LOSSES:
        loss_tops = [1.8, 2, 1.8, 3.5]
        loss_bots = [1.3, 1.4, None, None]
        loss_rights = [None, None, None, 50]
        loss_lefts = [None, None, None, -1]
        for (ctrl_model_type, bdnn_model_type), loss_top, loss_bot, loss_left, loss_right in zip(ALL_MODEL_PAIRS,
                                                                                                 loss_tops,
                                                                                                 loss_bots,
                                                                                                 loss_lefts,
                                                                                                 loss_rights):
            #show_graph("val_forward_f1s", LabelType.ALCO_STIMULUS, ctrl_model_type, bdnn_model_type)
            #show_graph("val_forward_accs", LabelType.ALCO_STIMULUS, ModelType.LENET5, ModelType.BLENET5)

            show_graph("val_forward_losses",
                       LabelType.ALCOHOLIC,
                       ctrl_model_type,
                       bdnn_model_type,
                       loss_top,
                       loss_bot,
                       loss_left,
                       loss_right)
    else:
        model_type_interest = [ModelType.BDNN, ModelType.BDNN_PCA, ModelType.BLENET5, ModelType.BRESNET56]
        for model_type in model_type_interest:
            plot_features(DatasetType.IMAGES, LabelType.ALCOHOLIC, model_type, 10)
    plt.show()


if __name__ == "__main__":
    main(GraphType.FEATURES)
    #torch.manual_seed(0)
    #plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.DNN_PCA, 0)
    #for flip_freq in ALL_FLIP_FREQS:
    #    plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.BDNN_PCA, flip_freq)
        # plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.BLENET5, 1)
        # plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.BLENET5, 10)
        # plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.BLENET5, 50)
        # plot_features(DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.BLENET5, 100)
    #print_best_accs("images_alcoholic")
    #show_graph("val_forward_f1s", LabelType.ALCOHOLIC, ModelType.DNN, ModelType.BDNN)
    #f = np.load("results/images_alcoholic/_CNN.npz")
    #print(float(dict(f)["test_f1"]))
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100_50"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100_50"])
