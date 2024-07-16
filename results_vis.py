#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torch import nn

from data import generate_dataset
from main import test
from models import get_model
from utils import get_results_path, DATA_ROOT, DEVICE, process_batch_forwards
from pacmap import PaCMAP
import torch
import pandas as pd

from models.configs import DatasetType, LabelType, ModelType, ForwardConfig
import bokeh.plotting as plt
from bokeh.palettes import Category10_4, Vibrant4, Colorblind4, Paired6

KEY_TO_TITLE = {
    "val_forward_accs": "Forward Val Accuracy vs. Epoch",
    "val_forward_f1s": "Forward Val F1 vs. Epoch",
    "val_forward_losses": "Forward Val Loss vs. Epoch",
    "val_backward_losses": "Backward Val Loss vs. Epoch",
}


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


def plot_features(dset_type: DatasetType, label_type: LabelType, model_type: ModelType, flip_freq: int):
    filepath = get_results_path(dset_type, label_type, model_type, flip_freq)
    state_dict = torch.load(filepath)["state_dict"]

    dl_train, dl_val, dl_test = generate_dataset(DATA_ROOT, dset_type, label_type, model_type, return_datasets=False)
    model = get_model(label_type, model_type).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    out_classes = label_type.out_classes()
    train_features = []
    train_labels = []
    train_correct = []
    with torch.no_grad():
        for batch, labels in dl_test:
            batch = batch.to(DEVICE)
            batch = process_batch_forwards(batch, model)
            train_features.append(model(batch, config=ForwardConfig.FEATURES_ONLY).cpu())
            outs = 2*model(batch, config=ForwardConfig.FORWARD).cpu().argmax(dim=1) // out_classes
            labels = 2*labels // out_classes
            correct: torch.Tensor = outs == labels
            train_correct.append(correct)
            train_labels.append(labels)
    train_features = torch.cat(train_features, dim=0).flatten(1)
    train_labels = torch.cat(train_labels).long()
    train_correct = torch.cat(train_correct).long()
    train_colours = 2 * train_labels + train_correct

    train_embed = PaCMAP()
    colours = [Paired6[5], Paired6[0], Paired6[1], Paired6[4]]
    train_colours = [colours[x.item()] for x in train_colours]
    train_features = torch.Tensor(train_embed.fit_transform(train_features.numpy()))

    # 0 = correct sober,
    df = {
        "train_features_x": train_features[:, 0] / torch.abs(train_features[:, 0]).max(),
        "train_features_y": train_features[:, 1] / torch.abs(train_features[:, 1]).max(),
        "train_labels": train_labels,
        "train_correct": train_correct,
        "train_colours": train_colours
    }
    df = pd.DataFrame(df)
    source = plt.ColumnDataSource.from_df(df)

    train_fig = plt.figure()
    train_fig.scatter("train_features_x", "train_features_y", source=source, color="train_colours")
    plt.show(train_fig)


if __name__ == "__main__":
    torch.manual_seed(0)
    """show_graph(DatasetType.IMAGES,
               LabelType.ALCOHOLIC,
               "val_forward_losses")"""
    plot_features(DatasetType.IMAGES, LabelType.ALCO_SUBJECTID, ModelType.LENET5, 1)
    #print_best_accs("images_alcoholic")
    #show_graph("images_alcoholic", "val_forward_f1s")
    #f = np.load("results/images_alcoholic/_CNN.npz")
    #print(float(dict(f)["test_f1"]))
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100"])
    #show_graph("stimulus_combined", "test_forward_accs", layers=["100_50"])
    #show_graph("stimulus_combined", "test_forward_losses", layers=["100_50"])