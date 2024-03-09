from argparse import ArgumentParser
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from src.entailment_tester import EntailmentTester
from src.utils import read_data


def main(args):
    print("Model:", args.model_name)

    pairs, labels, _ = read_data(args.pairs_file)

    tester = None
    if os.path.exists(args.output_file):
        scores_and_lens = pickle.load(open(args.output_file, "rb"))
    else:
        tester = EntailmentTester(
            args.model_name,
            revision=args.revision,
            lhs_repetition=args.lhs_repetition,
            rhs_repetition=args.rhs_repetition,
            tie_repetitions=args.tie_repetitions,
            repeat_whitespace=args.repeat_whitespace,
        )

        scores_and_lens = tester.test_entailment(*zip(*pairs), bsz=args.bsz)
        assert not os.path.exists(args.output_file)
        pickle.dump(scores_and_lens, open(args.output_file, "wb"))

    assert (
        all(probs is None for _, probs, _ in scores_and_lens)
        or all(len(probs) == 4 for _, probs, _ in scores_and_lens)
    )

    scores = [score for score, _, _ in scores_and_lens]
    report(scores, labels)


def report(scores, labels):
    pos_scores = [score for score, label in zip(scores, labels, strict=True) if label == "e"]
    neg_scores = [
        score for score, label in zip(scores, labels, strict=True) if label in {"nc", "n", "c"}
    ]
    assert len(pos_scores) + len(neg_scores) == len(scores)

    print(f"Avg pos score: {np.mean(pos_scores)}")
    print(f"Avg neg score: {np.mean(neg_scores)}")

    # Need entailment < non_entailment
    true = [0.0 for _ in pos_scores] + [1.0 for _ in neg_scores]
    pred = pos_scores + neg_scores
    auc = roc_auc_score(true, pred)
    print("ROC-AUC:", auc)
    print("inv ROC-AUC:", 1 - auc)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("pairs_file", type=str)
    parser.add_argument("output_file", type=str)  # the file used to pickle-dump scores
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--lhs_repetition", "-l", type=int, default=1)
    parser.add_argument("--rhs_repetition", "-r", type=int, default=1)
    parser.add_argument("--tie_repetitions", "-t", action="store_true")
    parser.add_argument("--repeat_whitespace", "-w", action="store_true")
    parser.add_argument("--bsz", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main(parse_args())  # pylint: disable=no-value-for-parameter
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
