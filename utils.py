from typing import Optional

import torch

from models import BDNN
from models.configs import FeatureModel, LabelType, ModelType, DatasetType, ForwardConfig
from pathlib import Path

# Device to use.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_ROOT = "results_new"
CHECKPOINTS_ROOT = "checkpoints"
DATA_ROOT = "C:/ml_datasets"
FIGURES_ROOT = "figures"

ALL_MODEL_PAIRS = [(ModelType.DNN, ModelType.BDNN),
                   (ModelType.DNN_PCA, ModelType.BDNN_PCA),
                   (ModelType.LENET5, ModelType.BLENET5),
                   (ModelType.RESNET56, ModelType.BRESNET56)]

ALL_MODEL_TYPES = [ModelType.DNN,
                   ModelType.BDNN,
                   ModelType.DNN_PCA,
                   ModelType.BDNN_PCA,
                   ModelType.LENET5,
                   ModelType.BLENET5,
                   ModelType.RESNET56,
                   ModelType.BRESNET56]

ALL_LABEL_TYPES = [LabelType.ALCOHOLIC,
                   LabelType.ALCO_STIMULUS,
                   LabelType.ALCO_SUBJECTID]

ALL_FLIP_FREQS = [1, 10, 50, 100]

class TrainingTracker:
    def __init__(self,
                 lambd: int,
                 model_type: ModelType,
                 label_type: LabelType,
                 flip_freq: Optional[int],
                 dataset_type: DatasetType = DatasetType.IMAGES,
                 checkpoint_root: str=CHECKPOINTS_ROOT):
        self.lambd = lambd
        self.model_type = model_type
        self.label_type = label_type
        self.dataset_type = dataset_type
        self.flip_freq = flip_freq
        self.checkpoint_root = checkpoint_root

        self.checkpoint_dir = Path(self.checkpoint_root)
        self.checkpoint_file = f"checkpoint.pt"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.reset()

    def reset(self):
        self.epoch_counter = 1
        self.checkpoint_epoch = -1
        self.is_forward_epochs = []

        self.train_forward_losses = []
        self.train_backward_losses = []

        self.val_foward_losses = []
        self.val_foward_accs = []
        self.val_forward_f1s = []
        self.val_backward_losses = []

        self.best_forward_acc = 0
        self.best_forward_f1 = 0
        self.best_forward_loss = float("inf")

    def update_forward(self, train_loss, val_loss, acc, f1, model: FeatureModel):
        self.train_forward_losses.append(train_loss)
        self.val_foward_losses.append(val_loss)
        self.val_foward_accs.append(acc)
        self.val_forward_f1s.append(f1)
        self.is_forward_epochs.append(True)

        self.best_forward_acc = max(self.best_forward_acc, acc)
        self.best_forward_f1 = max(self.best_forward_f1, f1)

        if val_loss < self.best_forward_loss:
            self.best_forward_loss = val_loss
            self.checkpoint_epoch = self.epoch_counter
            self.save(model, str(self.checkpoint_file))

        self.epoch_counter += 1

    def checkpoint_path(self):
        return self.checkpoint_dir / self.checkpoint_file

    def save(self, model, filename):
        torch.save(model.state_dict(), str(self.checkpoint_dir / filename))

    def update_backward(self, train_loss, val_loss):
        self.val_backward_losses.append(val_loss)
        self.train_backward_losses.append(train_loss)
        self.is_forward_epochs.append(False)
        self.epoch_counter += 1

    def get_forward(self):
        return {
            "val_acc": self.val_foward_accs[-1],
            "val_f1": self.val_forward_f1s[-1],
            "val_loss": self.val_foward_losses[-1],
            "train_loss": self.train_forward_losses[-1]
        }

    def get_backward(self):
        return {
            "back_train_loss": self.train_backward_losses[-1],
            "back_val_loss": self.val_backward_losses[-1]
        }

    def __lt__(self, other):
        assert other is not None
        assert isinstance(other, TrainingTracker)

        return self.best_forward_f1 < other.best_forward_f1

    def __le__(self, other):
        assert other is not None
        assert isinstance(other, TrainingTracker)

        return self.best_forward_f1 <= other.best_forward_f1

    def __str__(self):
        out = (f"Tracker Results:\n"
               f"Checkpoint epoch: {self.checkpoint_epoch}\n"
               f"Best val loss (->): {self.best_forward_loss}\n"
               f"Best val acc (->): {self.best_forward_acc}\n"
               f"Best val f1 (->): {self.best_forward_f1}\n"
               f"Best lambda: {self.lambd}")
        return out


def get_results_path(dset_version: DatasetType,
                     label_type: LabelType,
                     model_type: ModelType,
                     flip_freq: int):
    out_dir = Path(RESULTS_ROOT) / f"{dset_version}_{label_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"_F{flip_freq}" if model_type.is_bidirectional() else ""
    out_path = out_dir / f"{model_type}{prefix}.pt"
    return out_path


def process_batch_forwards(batch, model):
    if isinstance(model, BDNN):
        batch = batch.flatten(1)
    return batch


def process_batch_backwards(batch, model):
    """
    Convert the batch of images into features. For CNN models this means passing the images through the CNN layers only
    For regular BDNN models the image only needs to be flattened into a vector.
    :param batch:
    :param model:
    :return:
    """
    if isinstance(model, BDNN):
        batch = batch.flatten(1)
    else:
        batch = model(batch, config=ForwardConfig.FEATURES_ONLY)
    return batch
