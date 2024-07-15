from data import generate_dataset
import torch

from models.configs import DatasetType, LabelType
from utils import DATA_ROOT

if __name__ == "__main__":
    torch.manual_seed(0)
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC, pca_components=32*32*3, show_pca_debug=True)
    # prints 47
    generate_dataset(DATA_ROOT, DatasetType.IMAGES, LabelType.ALCOHOLIC, pca_components=47,
                     show_pca_debug=True)
    # also prints 47