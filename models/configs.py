from enum import Enum
from torch import nn


class ForwardConfig(Enum):
    FORWARD = 0
    BACKWARD = 1
    FEATURES_ONLY = 2


def check_backward_or_forward(c: ForwardConfig):
    assert c in {ForwardConfig.FORWARD, ForwardConfig.BACKWARD}


class ModelType(Enum):
    DNN = 0
    BDNN = 1
    DNN_PCA = 2
    BDNN_PCA = 3
    LENET5 = 4
    BLENET5 = 5
    RESNET56 = 6
    BRESNET56 = 7

    @classmethod
    def from_string(cls, name: str):
        names = {
            "DNN": cls.DNN,
            "BDNN": cls.BDNN,
            "DNN_PCA": cls.DNN_PCA,
            "BDNN_PCA": cls.BDNN_PCA,
            "LENET5": cls.LENET5,
            "BLENET5": cls.BLENET5,
            "RESNET56": cls.RESNET56,
            "BRESNET56": cls.BRESNET56
        }
        return names[name]

    def is_bidirectional(self):
        tags = [False, True] * 4
        return tags[self.value]

    def __str__(self):
        lst = ["DNN", "BDNN", "DNN_PCA", "BDNN_PCA", "LENET5", "BLENET5", "RESNET56", "BRESNET56"]
        return lst[self.value]


class FeatureModel(nn.Module):
    def __init__(self, forward_only: bool, out_features: int):
        """
        Constructor.
        :param forward_only: Represents whether the model should only be a forward model e.g: True on BDNN results in DNN.
        """
        super().__init__()
        self.forward_only = forward_only
        self.out_features = out_features

    def freeze_features(self):
        """
        Freezes all the feature layers of the model.
        :return:
        """
        pass

    def unfreeze_features(self):
        """
        Unfreezes all the feature layers of the model.
        :return:
        """
        pass


if __name__ == "__main__":
    print(ModelType.DNN.name)


class LabelType(Enum):
    ALCOHOLIC = 0
    ALCO_STIMULUS = 1
    ALCO_SUBJECTID = 2

    @classmethod
    def from_string(cls, name):
        names = {
            "ALCOHOLIC": cls.ALCOHOLIC,
            "ALCO_STIMULUS": cls.ALCO_STIMULUS,
            "ALCO_SUBJECTID": cls.ALCO_SUBJECTID
        }
        return names[name]

    def display(self):
        descs = ["Control", "Alcoholic Stimulus", "Alcoholic Subject"]
        return descs[self.value]

    def out_classes(self):
        outs = [2, 10, 244]
        return outs[self.value]

    def __str__(self):
        names = ["alcoholic", "alcoholic_stimulus", "alcoholic_subjectid"]
        return names[self.value]


class DatasetType(Enum):
    ORIGINAL = 0
    IMAGES = 1

    @classmethod
    def from_string(cls, name):
        names = {
            "ORIGINAL": cls.ORIGINAL,
            "IMAGES": cls.IMAGES
        }
        return names[name]

    def __str__(self):
        names = ["original", "images"]
        return names[self.value]
