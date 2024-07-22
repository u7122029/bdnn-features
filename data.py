from typing import Optional

import numpy as np
from bokeh.io import export_svgs
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
import scipy
from sklearn.decomposition import PCA
from models.configs import DatasetType, ModelType
from models import LabelType
from enum import Enum
#import bokeh.plotting as plt
import matplotlib.pyplot as plt

from utils import DATA_ROOT, FIGURES_ROOT


class PCAConfig(Enum):
    Kaiser = 0
    SCREE = 1
    RATIO_95 = 2


class AlcoholDataset(Dataset):
    def __init__(self,
                 X,
                 y_alcoholic,
                 y_stimulus,
                 subject_ids,
                 label_version: LabelType,
                 transform=None):
        super().__init__()

        self.X = X
        self.y_alcoholic = y_alcoholic
        self.y_stimulus = y_stimulus
        self.subject_ids = subject_ids
        self.label_version = label_version
        #self.trialnums = trialnums
        self.version = label_version
        self.transform = transform

        self.y = None
        self.y_alco_stim = 5 * self.y_alcoholic + self.y_stimulus - 1
        self.y_alco_sub = 122 * self.y_alcoholic + self.subject_ids - 1
        if label_version == LabelType.ALCOHOLIC:
            self.y = self.y_alcoholic
        elif label_version == LabelType.ALCO_STIMULUS:
            self.y = self.y_alco_stim
        elif label_version == LabelType.ALCO_SUBJECTID:
            self.y = self.y_alco_sub
        else:
            raise Exception("Invalid dataset type.")

        self.y = self.y.long()

    def __getitem__(self, item):
        out_x: torch.Tensor = self.X[item]
        out_y = self.y[item]
        if self.transform is not None:
            ndims = out_x.dim()
            out_x = out_x.numpy()
            if ndims == 4:
                out_x = torch.stack([self.transform(img) for img in out_x])
            else:
                out_x = self.transform(out_x)
        return out_x, out_y

    def __len__(self):
        return len(self.X)


def apply_pca(X_train, X_val, pca_components):
    train_val_comb = torch.concat([X_train, X_val], dim=0)
    pca = PCA(n_components=pca_components)
    pca.fit(train_val_comb.flatten(1))
    return pca


def generate_dataset(data_root: str,
                     version: DatasetType,
                     label_version: LabelType,
                     model_type: ModelType,
                     train_prop=0.7,
                     val_prop=0.1,
                     train_batch_size=64,
                     return_datasets=False,
                     pca_components: Optional[int]=None,
                     get_pca_results: Optional[PCAConfig]=None):
    """
    Generates a variant of the UCI EEG Alcoholism dataset based on specifications.
    :param data_root: The root directory of the raw dataset.
    :param version:
    :param label_version: The variant to generate.
    :param model_type:
    :param train_prop: The proportion of the raw data that should be in the training set.
    :param val_prop: The proportion of the raw data that should be in the validation set.
    :param train_batch_size: The batch size for the training set.
    :param return_datasets: True, if the datasets should be returned instead of the dataloaders.
    :param pca_components:
    :param get_pca_results:
    :return: The training and testing datasets or dataloaders based on return_datasets.
    """
    if model_type in (ModelType.DNN_PCA, ModelType.BDNN_PCA) and pca_components is None:
        pca_components = 47

    if version == DatasetType.ORIGINAL:
        raw_dataset = scipy.io.loadmat(str(Path(data_root) / "alcoholism" / "uci_eeg.mat"))
        X = torch.Tensor(raw_dataset["X"])
    elif version == DatasetType.IMAGES:
        raw_dataset = scipy.io.loadmat(str(Path(data_root) / "alcoholism_v2" / "uci_eeg_images.mat"))
        X = torch.Tensor(raw_dataset["data"])
    else:
        raise Exception(f"Invalid version '{version}'.")

    y_alcoholic = torch.Tensor(raw_dataset["y_alcoholic"]).squeeze(0)
    y_stimulus = torch.Tensor(raw_dataset["y_stimulus"]).squeeze(0)
    subject_ids = torch.Tensor(raw_dataset["subjectid"]).squeeze(0)

    """if version == DatasetType.ORIGINAL:
        # Extract the theta, alpha and beta EEG layers.
        X_theta = X[:, 4:8, :]
        X_theta = torch.mean(X_theta, dim=1).unsqueeze(1)
        X_alpha = X[:, 8:14, :]
        X_alpha = torch.mean(X_alpha, dim=1).unsqueeze(1)
        X_beta = X[:, 14:31, :]
        X_beta = torch.mean(X_beta, dim=1).unsqueeze(1)
        X = torch.cat([X_theta, X_alpha, X_beta], dim=1)
        X = X.flatten(start_dim=1)"""
    # If the dataset version consists of images, then all the preprocessing has been done for us.

    # Shuffle
    shuffle_indices = torch.randperm(len(X))
    X = X[shuffle_indices, :]
    y_alcoholic = y_alcoholic[shuffle_indices]
    y_stimulus = y_stimulus[shuffle_indices]
    subject_ids = subject_ids[shuffle_indices]

    # Train val test split.
    pivot = int(len(X) * train_prop)
    val_offset = int(len(X) * val_prop)

    X_train = X[:pivot]
    X_val = X[pivot:pivot + val_offset]
    X_test = X[pivot + val_offset:]

    y_alcoholic_train = y_alcoholic[:pivot]
    y_alcoholic_val = y_alcoholic[pivot:pivot + val_offset]
    y_alcoholic_test = y_alcoholic[pivot + val_offset:]

    y_stimulus_train = y_stimulus[:pivot]
    y_stimulus_val = y_stimulus[pivot:pivot + val_offset]
    y_stimulus_test = y_stimulus[pivot + val_offset:]

    subject_ids_train = subject_ids[:pivot]
    subject_ids_val = subject_ids[pivot:pivot + val_offset]
    subject_ids_test = subject_ids[pivot + val_offset:]

    transform = None
    if version == DatasetType.IMAGES:
        # Normalise by converting each pixel to the interval [0,1], then standardising to mean 0, std 1
        transform = Compose([ToTensor(),
                             Normalize([0.40694301044590, 0.46442207869852, 0.40599444799447],
                                       [0.42642453394862, 0.48742047883986, 0.42334384481327])
                             ])

    if pca_components is not None:
        transform = None
        train_val_comb = torch.concat([X_train, X_val], dim=0)
        m, s = (train_val_comb.mean(), train_val_comb.std())
        X_train = (X_train - m) / s
        X_val = (X_val - m) / s
        X_test = (X_test - m) / s

        pca = apply_pca(X_train, X_val, pca_components)
        if get_pca_results == PCAConfig.Kaiser:
            # Perform Kaiser's Rule to determine how many components to use.
            eigenvalues = pca.explained_variance_
            components_to_keep = sum([1 for eigenvalue in eigenvalues if eigenvalue > 1])
            print(components_to_keep)
        elif get_pca_results == PCAConfig.RATIO_95:
            cs = np.cumsum(pca.explained_variance_ratio_)
            indices = np.where(cs >= 0.95)[0]
            print(indices[0])
        elif get_pca_results == PCAConfig.SCREE:
            # Get the explained variance
            explained_variance = pca.explained_variance_
            xs = list(range(1, len(explained_variance) + 1))

            plt.figure()
            plt.bar(xs, explained_variance, color="#ffc8bd")
            plt.plot(xs, explained_variance)
            plt.plot([0,len(explained_variance) + 1], [1, 1], color="red", label="Kaiser Thresh.")
            plt.plot([47, 47], [5, 0], color="green", label="Kaiser Thresh. Intersection")
            plt.ylim(0, 5)
            plt.legend(loc="best")
            plt.title("Scree Plot of Principal Components")
            plt.ylabel("Eigenvalue Magnitude")
            plt.xlabel("Principal Component")
            plt.savefig(f"{FIGURES_ROOT}/scree.svg", format="svg")
            plt.show()

        X_train = torch.Tensor(pca.transform(X_train.flatten(1)))
        X_val = torch.Tensor(pca.transform(X_val.flatten(1)))
        X_test = torch.Tensor(pca.transform(X_test.flatten(1)))

    # Put processed data into dataset splits.

    dset_train = AlcoholDataset(X_train, y_alcoholic_train, y_stimulus_train, subject_ids_train, label_version,
                                transform=transform)
    dset_val = AlcoholDataset(X_val, y_alcoholic_val, y_stimulus_val, subject_ids_val, label_version,
                              transform=transform)
    dset_test = AlcoholDataset(X_test, y_alcoholic_test, y_stimulus_test, subject_ids_test, label_version,
                               transform=transform)

    if return_datasets:
        return dset_train, dset_val, dset_test

    # Generate training and testing dataloaders.
    if train_batch_size == "all":
        train_batch_size = len(dset_train)

    dataloader_train = DataLoader(dset_train, batch_size=train_batch_size, shuffle=True)
    dataloader_val = DataLoader(dset_val, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dset_test, batch_size=64)
    return dataloader_train, dataloader_val, dataloader_test


if __name__ == "__main__":
    torch.manual_seed(0)
    dset_train, dset_val, dset_test = generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCO_STIMULUS, ModelType.DNN, return_datasets=True)
    y1, y2, y3 = dset_test.y_alcoholic, 2*dset_test.y_alco_stim // 10, 2*dset_test.y_alco_sub // 244
    #print(dset_test.subject_ids.max())
    #print(dset_train.subject_ids.max())
    #print(dset_val.subject_ids.max())
    #print(torch.all(y1 - y2 == 0))
    #print(torch.all(y2 - y3 == 0))