import torch

from models.configs import DatasetType, LabelType
from utils import get_results_path
import pandas as pd
from utils import ALL_MODEL_PAIRS, ALL_LABEL_TYPES, ALL_FLIP_FREQS


def generate_df(acc_key, f1_key):
    accs = {}
    f1s = {}
    results = {
        "label_type": [LabelType.ALCOHOLIC] * 5 + [LabelType.ALCO_STIMULUS] * 5 + [LabelType.ALCO_SUBJECTID] * 5,
        "flip_freq": [0, 1, 10, 50, 100] * 3
    }
    for label_type in ALL_LABEL_TYPES:
        for original_model_type, bd_model_type in ALL_MODEL_PAIRS:
            orig_model_filepath = get_results_path(DatasetType.IMAGES, label_type, original_model_type, 0)
            data = torch.load(orig_model_filepath)
            if bd_model_type.name not in accs:
                accs[bd_model_type.name] = []

            if bd_model_type.name not in f1s:
                f1s[bd_model_type.name] = []

            f1s[bd_model_type.name].append(data[acc_key])
            accs[bd_model_type.name].append(data[f1_key])

            for flip_freq in ALL_FLIP_FREQS:
                bd_model_filepath = get_results_path(DatasetType.IMAGES, label_type, bd_model_type, flip_freq)
                data = torch.load(bd_model_filepath)

                f1s[bd_model_type.name].append(data[acc_key])
                accs[bd_model_type.name].append(data[f1_key])

                # test_results["label_type"].append(label_type)
    df_f1s = pd.concat({
        "": pd.DataFrame(results),
        f1_key: pd.DataFrame(f1s).apply(lambda x: round(x, 3))
    }, axis=1, names=["", ""])
    df_accs = pd.concat({
        "": pd.DataFrame(results),
        acc_key: pd.DataFrame(accs).apply(lambda x: round(x, 3)),
    }, axis=1, names=["", ""])
    return df_accs, df_f1s


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 3)
    test_acc_df, test_f1_df = generate_df("test_acc", "test_f1")
    val_acc_df, val_f1_df = generate_df("best_val_acc", "best_val_f1")
    print(test_acc_df.to_latex(index=False, float_format="%.3f"))
    print(test_f1_df.to_latex(index=False, float_format="%.3f"))
    #print(test_df.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()