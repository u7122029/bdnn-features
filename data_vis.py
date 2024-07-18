from utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import generate_dataset


def generate_plot(dset_version: DatasetType = DatasetType.IMAGES, label_version=LabelType.ALCOHOLIC):
    dset_train, dset_val, dset_test = generate_dataset("C:/ml_datasets",
                                                       version=dset_version,
                                                       label_version=label_version,
                                                       model_type=ModelType.DNN,
                                                       train_batch_size=64,
                                                       return_datasets=True)
    full_x = torch.cat([dset_train.X, dset_val.X, dset_test.X])
    full_y_alcoholic = torch.cat([dset_train.y_alcoholic, dset_val.y_alcoholic, dset_test.y_alcoholic])
    full_y_stimulus = torch.cat([dset_train.y_stimulus, dset_val.y_stimulus, dset_test.y_stimulus])
    full_subjectids = torch.cat([dset_train.subject_ids, dset_val.subject_ids, dset_test.subject_ids]).int()
    full_subjectids, _ = torch.sort(torch.bincount(full_subjectids), descending=True)

    # All distributions shown below are after preprocessing.
    # Show distribution of X
    plt.figure()
    plt.hist(full_x.flatten().numpy(), bins=700)
    plt.title("Distribution of X")
    if dset_version == DatasetType.ORIGINAL:
        plt.xlim([-75, 70])
    else:
        plt.xlim([-0.5, 2.5])

    plt.ylabel("freq")
    plt.xlabel("intensity")
    plt.savefig(f"figures/X_images_unprocessed.svg", format="svg")

    # Show distribution of y_alcoholic classes
    plt.figure()
    plt.bar(["0", "1"], [torch.sum((full_y_alcoholic == 0).long()).item(),
                         torch.sum((full_y_alcoholic == 1).long()).item()])
    plt.title("y_alcoholic Distribution")
    plt.ylabel("freq")
    plt.savefig(f"figures/y_alco_images_unprocessed.svg", format="svg")

    # Show distribution of y_stimulus classes
    lst_y_stimulus = [torch.sum((full_y_stimulus == i).long()).item() for i in range(1, 6)]
    plt.figure()
    plt.bar(["1", "2", "3", "4", "5"], lst_y_stimulus)
    plt.title("y_stimulus Distribution")
    plt.ylabel("freq")
    plt.savefig(f"figures/y_stim_images_unprocessed.svg", format="svg")

    # Show distribution of subject_ids
    plt.figure()
    plt.bar(list(range(1, 1+len(full_subjectids))), full_subjectids.numpy())
    plt.title("subjectid Distribution (Sorted by Freq.)")
    plt.ylabel("freq")
    plt.xlabel("subjectid Index")
    plt.savefig(f"figures/subjectid_images_unprocessed.svg", format="svg")

    plt.figure()
    img = full_x[0]
    img /= img.max()
    print(f"Label: {full_y_alcoholic[0]}")
    plt.imshow(img)
    plt.savefig(f"figures/sampleimg_images_unprocessed.svg", format="svg")


if __name__ == "__main__":
    #show_dataset_pca(DSETV_IMAGES, LBLV_UNPROCESSED)
    generate_plot()
    #for dset_type in ["unprocessed", "original"]:
    #    generate_plot(dset_type)
