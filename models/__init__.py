from .bdnn import BDNN
from .configs import ModelType, LabelType
from .lenet5_bdnn import LeNet5_BDNN
from .resnet_bdnn import ResNet_BDNN, resnet56, resnet20


def get_model(label_version: LabelType, model_type: ModelType):
    """
    Gets the model based on specifications.
    :param label_version: The label version of the dataset.
    :param model_type: The model type
    :return: The model.
    """
    out_classes = label_version.out_classes()
    if model_type == ModelType.DNN:
        return BDNN(3072, out_classes, True, [120, 84])
    elif model_type == ModelType.BDNN:
        return BDNN(3072, out_classes, False, [120, 84])
    elif model_type == ModelType.DNN_PCA:
        return BDNN(47, out_classes, True, [120, 84])
    elif model_type == ModelType.BDNN_PCA:
        return BDNN(47, out_classes, False, [120, 84])
    elif model_type == ModelType.LENET5:
        return LeNet5_BDNN(out_classes, True)
    elif model_type == ModelType.BLENET5:
        return LeNet5_BDNN(out_classes, False)
    elif model_type == ModelType.RESNET56:
        return resnet56(out_classes, True)
    else:
        return resnet56(out_classes, False)