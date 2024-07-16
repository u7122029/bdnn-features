from data import generate_dataset, PCAConfig
import torch

from models.configs import DatasetType, LabelType
from utils import DATA_ROOT


def kaiser_guttman_rule():
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC,
                     pca_components=32 * 32 * 3, get_pca_results=PCAConfig.Kaiser)
    # prints 47
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC,
                     pca_components=47, get_pca_results=PCAConfig.Kaiser)


def scree():
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC,
                     pca_components=60, get_pca_results=PCAConfig.SCREE)


if __name__ == "__main__":
    torch.manual_seed(0)
    scree()