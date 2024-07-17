import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from data import generate_dataset
from models import get_model
from utils import *
import torch.nn.functional as F
from pathlib import Path
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall, BinaryPrecision
from torchmetrics import MeanMetric
import json
import shutil

from models.configs import FeatureModel, ForwardConfig, DatasetType, check_backward_or_forward
from utils import get_results_path, process_batch_forwards, process_batch_backwards


def test(model: FeatureModel, test_dataloader, criterion, config: ForwardConfig, return_pr=False):
    """
    Tests a model over the test dataloader.
    :param model: The model
    :param test_dataloader: The test dataloader.
    :param criterion: The criterion
    :param config:
    :return: The average loss and acc - nonexistent for the backwards direction.
    """
    assert check_backward_or_forward(config)

    num_classes = model.out_features
    precisionMetric = BinaryPrecision()
    recallMetric = BinaryRecall()
    f1Metric = BinaryF1Score()
    accuracyMetric = BinaryAccuracy()
    meanLoss = MeanMetric()

    precisionMetric.to(DEVICE)
    recallMetric.to(DEVICE)
    f1Metric.to(DEVICE)
    accuracyMetric.to(DEVICE)
    meanLoss.to(DEVICE)

    if model.training:
        after_test = lambda: model.train()
    else:
        after_test = lambda: model.eval()

    model.eval()
    mean_loss = None
    acc = None
    f1 = None
    precision = None
    recall = None
    with torch.no_grad():
        for batch, labels in test_dataloader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            if config == ForwardConfig.BACKWARD:
                # Pass the batch through the convolutional layers, if there are any.
                batch = process_batch_backwards(batch, model)

                # one hot encode labels to pass through output layer.
                labels = F.one_hot(labels, num_classes).float()
                outs = model(labels, config=config)
                loss = criterion(outs, batch)
            else:
                # Perform feedforward pass to compute loss.
                batch = process_batch_forwards(batch, model)
                outs = model(batch, config=config)
                loss = criterion(outs, labels)
            meanLoss.update(loss)

            if config == ForwardConfig.FORWARD:
                # Convert labels and outputs to perform binary classification.
                bin_preds = (torch.argmax(outs, dim=1) * 2) // num_classes
                bin_labels = (labels * 2) // num_classes
                accuracyMetric.update(bin_preds, bin_labels)
                f1Metric.update(bin_preds, bin_labels)
                precisionMetric.update(bin_preds, bin_labels)
                recallMetric.update(bin_preds, bin_labels)

        if config == ForwardConfig.FORWARD:
            acc = accuracyMetric.compute().item()
            f1 = f1Metric.compute().item()
            precision = precisionMetric.compute().item()
            recall = recallMetric.compute().item()
        mean_loss = meanLoss.compute().item()

    after_test()
    if return_pr:
        return mean_loss, acc, f1, precision, recall
    return mean_loss, acc, f1


def epoch_forwards(model: FeatureModel, train_dataloader, criterion, optimiser, lambd):
    """
    Performs an epoch on the model in the forwards direction.
    :param model: The model.
    :param train_dataloader: The training dataloader.
    :param criterion: The criterion.
    :param optimiser: The optimiser.
    :param lambd: The L2 regularisation coefficient/parameter.
    :return: The average loss over the entire test set.
    """
    meanLoss = MeanMetric()
    meanLoss.to(DEVICE)
    for batch, labels in train_dataloader:
        batch = batch.to(DEVICE)
        batch = process_batch_forwards(batch, model)
        labels = labels.to(DEVICE)

        optimiser.zero_grad()
        outs = model(batch)
        outs = outs.to(DEVICE)
        # regl1 = sum([torch.abs(param).sum() for param in model.parameters()])
        regl2 = sum([0.5 * (param ** 2).sum() for param in model.parameters()])
        loss = criterion(outs, labels) + (lambd * regl2)
        meanLoss.update(loss)

        loss.mean().backward()
        optimiser.step()

    return meanLoss.compute().item()  # average loss.


def epoch_backwards(model: FeatureModel, train_dataloader, criterion, optimiser):
    """
    Performs a single epoch in the backwards direction.
    :param model: The model.
    :param train_dataloader: The training dataloader.
    :param criterion: The criterion to compute backwards loss.
    :param optimiser: The optimiser.
    :return: The average loss over the entire epoch.
    """
    meanLoss = MeanMetric()
    meanLoss.to(DEVICE)
    num_classes = model.out_features

    model.freeze_features()

    for batch, labels in train_dataloader:
        batch = batch.to(DEVICE)
        batch = process_batch_backwards(batch, model)
        labels = labels.to(DEVICE)
        labels = F.one_hot(labels, num_classes=num_classes).float()

        optimiser.zero_grad()
        outs = model(labels, config=ForwardConfig.BACKWARD)
        outs = outs.to(DEVICE)
        # regl1 = sum([torch.abs(param).sum() for param in model.parameters()])
        # regl2 = sum([0.5 * (param ** 2).sum() for param in model.parameters()])
        loss = criterion(outs, batch)  # No regularisation for MSE
        meanLoss.update(loss)
        loss.mean().backward()
        optimiser.step()

    model.unfreeze_features()

    return meanLoss.compute().item()


def get_last_maybe(lst):
    if not lst:
        return None, None
    return lst[-1]


def train(model: FeatureModel,
          train_dataloader,
          val_dataloader,
          criterion_forwards,
          criterion_backwards,
          optimiser,
          epochs: int,
          tracker: TrainingTracker,
          flip_freq=50):
    """
    Trains a given model over a training dataloader and tests performance on testing dataloader.
    :param model: The model.
    :param train_dataloader: The dataloader for training.
    :param val_dataloader: The dataloader for testing.
    :param criterion_forwards: The criterion for outputs from the forward direction of the model.
    :param criterion_backwards: The criterion for outputs from the backward direction of the model.
    :param optimiser: The optimiser.
    :param epochs: The maximum number of epochs
    :param lambd: The L2 regularisation coefficient.
    :param flip_freq: The frequency that the model should flip its x direction (per epoch).
    :return: The best test set accuracy over the entire training process, as well as the losses and accuracies achieved
    during training.
    """
    lambd = tracker.lambd
    model.train()
    feed_forward = True

    progressbar = tqdm(range(1, epochs + 1), desc="Epoch")
    for epoch in progressbar:
        if feed_forward:
            train_loss_forward = epoch_forwards(model, train_dataloader, criterion_forwards, optimiser, lambd)
            val_loss_forward, val_acc_forward, val_f1_forward = test(model,
                                                                     val_dataloader,
                                                                     criterion_forwards,
                                                                     ForwardConfig.FORWARD)

            tracker.update_forward(train_loss_forward, val_loss_forward, val_acc_forward, val_f1_forward, model)
            postfix = tracker.get_forward()
        else:
            train_loss_backward = epoch_backwards(model, train_dataloader, criterion_backwards, optimiser)

            val_loss_backward, _, _ = test(model,
                                           val_dataloader,
                                           criterion_backwards,
                                           ForwardConfig.BACKWARD)
            tracker.update_backward(train_loss_backward, val_loss_backward)
            postfix = tracker.get_backward()

        progressbar.set_postfix(postfix)
        if not model.forward_only and epoch % flip_freq == 0:
            # Switch training direction.
            feed_forward = not feed_forward

    tqdm.write("Training done.")
    return tracker


def get_best_regularisation(train_dl,
                            val_dl,
                            model_type,
                            label_type,
                            lr,
                            lambdas,
                            flip_freq,
                            epochs):
    print("Getting best regularisation lambda.")
    criterion_forwards = nn.CrossEntropyLoss(reduction="none")
    criterion_backwards = nn.MSELoss(reduction="none")
    best_tracker = TrainingTracker(-1, model_type, label_type, flip_freq)

    for lambd in tqdm(lambdas, desc="Iterating through lambdas."):
        model = get_model(label_type, model_type)
        model.to(DEVICE)
        optimiser = optim.Adam(model.parameters(), lr=lr)
        tracker = TrainingTracker(lambd, model_type, label_type, flip_freq)
        train(model,
              train_dl,
              val_dl,
              criterion_forwards,
              criterion_backwards,
              optimiser,
              epochs,
              tracker,
              flip_freq=flip_freq)

        # So right now the best model state_dict for the current lambda has been saved.
        checkpoint_file_path = tracker.checkpoint_path()
        best_file_path = checkpoint_file_path.parent / "best_lambda.pt"
        if best_tracker < tracker:
            best_tracker = tracker
            shutil.copy(checkpoint_file_path, best_file_path)

    return best_tracker


def run_model(model_type: ModelType,
              label_type: LabelType,
              dset_version: DatasetType = DatasetType.IMAGES,
              root=DATA_ROOT,
              batch_size=64,
              epochs=449,
              flip_freq=50,
              lr=2e-5,
              test_only=False):
    """
    Generates a model and a dataset based on the given specifications, and trains the model on the dataset.
    :param root: The root directory of the original dataset.
    :param model_type:
    :param batch_size: The size of each batch in a dataloader.
    :param epochs: The maximum number of epochs to train for.
    :param flip_freq: The frequency that the model should flip its x direction (per epoch).
    :param dset_version: The dataset version to use.
    :param lr: The learning rate.
    :param label_type: The variant of the dataaset that should be generated.
    :return: None. Saves results to the results/ directory.
    """
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]

    train_dl, val_dl, test_dl = generate_dataset(root,
                                                 dset_version,
                                                 label_type,
                                                 model_type,
                                                 train_batch_size=batch_size)

    best_results = get_best_regularisation(train_dl,
                                           val_dl,
                                           model_type,
                                           label_type,
                                           lr,
                                           lambdas,
                                           flip_freq,
                                           epochs)

    model = get_model(label_type, model_type).to(DEVICE)
    model.load_state_dict(torch.load(best_results.checkpoint_dir / "best_lambda.pt"))

    _, test_acc, test_f1, test_prec, test_recall = test(model,
                                                        test_dl,
                                                        nn.CrossEntropyLoss(reduction="none"),
                                                        ForwardConfig.FORWARD,
                                                        return_pr=True)

    print(best_results)
    print(f"Test Acc: {test_acc}")
    print(f"Test F1: {test_f1}")
    print(f"Test Precision: {test_prec}")
    print(f"Test Recall: {test_recall}")

    out_path = get_results_path(dset_version, label_type, model_type, flip_freq)
    d = {"is_forward_epochs": best_results.is_forward_epochs,
         "best_val_acc": best_results.best_forward_acc,
         "best_val_f1": best_results.best_forward_f1,
         "val_forward_losses": best_results.val_foward_losses,
         "val_forward_accs": best_results.val_foward_accs,
         "val_forward_f1s": best_results.val_forward_f1s,
         "test_acc": test_acc,
         "test_f1": test_f1,
         "test_prec": test_prec,
         "test_recall": test_recall,
         'best_lambda': best_results.lambd,
         "state_dict": model.state_dict()}

    torch.save(d, str(out_path))


def main():
    torch.manual_seed(0)
    print(f"device: {DEVICE}")

    # Load config json file.
    config_dict = json.load(open("config.json"))
    lr = float(config_dict["lr"])
    epochs = int(config_dict["epochs"])
    dset_root = str(config_dict["dset_root"])
    if not Path(dset_root).exists():
        raise FileNotFoundError(f"Provided dataset root {dset_root} does not exist.")

    model_type_combos = [ModelType.DNN,
                         ModelType.BDNN,
                         ModelType.DNN_PCA,
                         ModelType.BDNN_PCA,
                         ModelType.LENET5,
                         ModelType.BLENET5,
                         ModelType.RESNET56,
                         ModelType.BRESNET56]
    dataset_label_version_combos = [LabelType.ALCOHOLIC, LabelType.ALCO_STIMULUS, LabelType.ALCO_SUBJECTID]
    flip_freq_combos = [1, 10, 50, 100]

    # Iterate over every type of model.
    for model_type in model_type_combos:
        for dset_label_version in dataset_label_version_combos:
            print(model_type, dset_label_version)

            if not model_type.is_bidirectional():
                # If the model type is not bidirectional we can run the model with no flip frequency and then
                # skip over to the next one.
                path = get_results_path(DatasetType.IMAGES, dset_label_version, model_type, flip_freq=0)
                if path.exists():
                    continue

                run_model(model_type,
                          dset_label_version,
                          flip_freq=0,
                          lr=lr,
                          epochs=epochs)
                continue

            for flip_freq in flip_freq_combos:
                print(f"Current flip frequency: {flip_freq}")
                path = get_results_path(DatasetType.IMAGES, dset_label_version, model_type, flip_freq=flip_freq)
                if path.exists():
                    continue
                run_model(model_type,
                          dset_label_version,
                          flip_freq=flip_freq,
                          lr=lr,
                          epochs=epochs)


if __name__ == "__main__":
    main()
