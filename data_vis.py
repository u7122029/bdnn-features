from utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch


from data import generate_dataset


def generate_plot(dset_version, label_version=LBLV_UNPROCESSED):
    dset_train, dset_val, dset_test = generate_dataset("C:/ml_datasets",
                                             version=dset_version,
                                             label_version=label_version,
                                             model_version=MDL_LINEAR,
                                             train_prop=0.7,
                                             val_prop = 0.1,
                                             train_batch_size=64,
                                             return_datasets=True,
                                             pca_components=None,
                                             undersample=False)
    full_x = torch.cat([dset_train.X, dset_val.X, dset_test.X])
    full_y_alcoholic = torch.cat([dset_train.y_alcoholic, dset_val.y_alcoholic, dset_test.y_alcoholic])
    full_y_stimulus = torch.cat([dset_train.y_stimulus, dset_val.y_stimulus, dset_test.y_stimulus])
    full_subjectids = torch.cat([dset_train.subject_ids, dset_val.subject_ids, dset_test.subject_ids])

    # All distributions shown below are after preprocessing.
    # Show distribution of X
    dsetv = DSETV_TO_STR[dset_version]
    lblv = LBLV_TO_STR[label_version]
    print(f"{dsetv}_{lblv}")

    plt.figure()
    plt.hist(full_x.flatten().numpy(), bins=700)
    plt.title("Distribution of X")
    if dset_version == DSETV_ORIGINAL:
        plt.xlim([-75,70])
    else:
        plt.xlim([-0.5,2.5])

    plt.ylabel("freq")
    plt.xlabel("intensity")
    plt.savefig(f"figures/X_{dsetv}_{lblv}.svg", format="svg")

    # Show distribution of y_alcoholic classes
    plt.figure()
    plt.bar(["0","1"], [torch.sum((full_y_alcoholic == 0).long()).item(),
                        torch.sum((full_y_alcoholic == 1).long()).item()])
    plt.title("y_alcoholic Distribution")
    plt.ylabel("freq")
    plt.savefig(f"figures/y_alco_{dsetv}_{lblv}.svg", format="svg")

    # Show distribution of y_stimulus classes
    lst_y_stimulus = [torch.sum((full_y_stimulus == i).long()).item() for i in range(1,6)]
    plt.figure()
    plt.bar(["1","2","3","4","5"], lst_y_stimulus)
    plt.title("y_stimulus Distribution")
    plt.ylabel("freq")
    plt.savefig(f"figures/y_stim_{dsetv}_{lblv}.svg", format="svg")

    # Show distribution of subject_ids
    plt.figure()
    plt.hist(full_subjectids.numpy(), bins=200)
    plt.title("subjectid Distribution")
    plt.ylabel("freq")
    plt.xlabel("subjectid")
    plt.savefig(f"figures/subjectid_{dsetv}_{lblv}.svg", format="svg")

    plt.figure()
    img = full_x[0]
    img /= img.max()
    print(f"Label: {full_y_alcoholic[0]}")
    plt.imshow(img)
    plt.savefig(f"figures/sampleimg_{dsetv}_{lblv}.svg", format="svg")

    #plt.show()

def show_dataset_pca(dset_version, label_version):
    """
    Shows the dataset images as 3D points after an application of PCA.
    """
    dset_train, dset_test = generate_dataset("C:/ml_datasets",
                                             version=dset_version,
                                             label_version=label_version,
                                             return_datasets=True,
                                             pca_components=3,
                                             undersample=False)
    X = dset_train.X

    positive = dset_train.y == 1
    negative = dset_train.y == 0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[positive, 0], X[positive, 1], X[positive, 2], marker=".", label="positive")
    ax.scatter(X[negative, 0], X[negative, 1], X[negative, 2], marker="x", label="negative")
    ax.set_title(f"PCA Decomposition of Dataset - {DSETV_TO_STR[dset_version]}, {LBLV_TO_STR[label_version]}")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.show()


if __name__ == "__main__":
    #show_dataset_pca(DSETV_IMAGES, LBLV_UNPROCESSED)
    generate_plot(DSETV_IMAGES, LBLV_UNPROCESSED)
    #for dset_type in ["unprocessed", "original"]:
    #    generate_plot(dset_type)