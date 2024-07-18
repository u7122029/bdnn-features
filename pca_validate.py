from data import generate_dataset, PCAConfig
import torch

from models.configs import DatasetType, LabelType, ModelType
from utils import DATA_ROOT


def kaiser_guttman_rule():
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC, ModelType.DNN_PCA,
                     pca_components=32 * 32 * 3, get_pca_results=PCAConfig.Kaiser)
    # prints 47


def scree():
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC, ModelType.DNN_PCA,
                     pca_components=200, get_pca_results=PCAConfig.SCREE)


def ratio_95():
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC, ModelType.DNN_PCA,
                     pca_components=100, get_pca_results=PCAConfig.RATIO_95) # prints 24


if __name__ == "__main__":
    torch.manual_seed(0)
    #kaiser_guttman_rule()
    #ratio_95()
    scree()
