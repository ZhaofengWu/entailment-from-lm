import os
import sys

import datasets
from tqdm import tqdm


DATASETS = {
    "rte": {
        "dataset_name": "glue",
        "subset_name": "rte",
        "premise_key": "sentence1",
        "hypothesis_key": "sentence2",
        "label_key": "label",
        "entailment_label": 0,
    },
    "mnli": {
        "dataset_name": "glue",
        "subset_name": "mnli",
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        "entailment_label": 0,
        "neutral_label": 1,
        "contradiction_label": 2,
    },
    "anli_r3": {
        "dataset_name": "anli",
        "subset_name": None,
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "label",
        "entailment_label": 0,
        "neutral_label": 1,
        "contradiction_label": 2,
    },
    "wanli": {
        "dataset_name": "alisawuffles/WANLI",
        "subset_name": None,
        "premise_key": "premise",
        "hypothesis_key": "hypothesis",
        "label_key": "gold",
        "entailment_label": "entailment",
        "neutral_label": "neutral",
        "contradiction_label": "contradiction",
    },
}


def main(dataset_name, split_name, output_file):
    assert dataset_name in DATASETS
    assert not os.path.exists(output_file)

    dataset_info = DATASETS[dataset_name]
    kwargs = {}
    dataset = datasets.load_dataset(
        dataset_info["dataset_name"], dataset_info["subset_name"], **kwargs
    )
    if "neutral_label" not in dataset_info:
        label_map = {
            dataset_info["entailment_label"]: "e", 1 - dataset_info["entailment_label"]: "nc"
        }
    else:
        label_map = {
            dataset_info["entailment_label"]: "e",
            dataset_info["neutral_label"]: "n",
            dataset_info["contradiction_label"]: "c",
        }


    with open(output_file, "w") as f:
        if dataset_name == "anli_r3":
            split_name += "_r3"
        for ex in tqdm(dataset[split_name]):

            def escape(s):
                # https://stackoverflow.com/questions/15392730/in-python-is-it-possible-to-escape-newline-characters-when-printing-a-string
                return s.encode("unicode_escape").decode("utf-8")

            f.write(
                f"{escape(ex[dataset_info['premise_key']])}"
                f"\t{escape(ex[dataset_info['hypothesis_key']])}"
                f"\t{label_map[ex[dataset_info['label_key']]]}\n"
            )


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
