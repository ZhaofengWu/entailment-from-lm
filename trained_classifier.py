import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils import read_data

np.set_printoptions(linewidth=1000)


def featurize(scores_and_lens):
    X = []
    for _, probs, _ in scores_and_lens:
        assert len(probs) == 4
        X.append(probs)
    return X


def map_labels(labels):
    map = {"e": 0, "nc": 1, "n": 1, "c": 2}
    return [map[label] for label in labels]


def main(data_file, scores_file):
    _, labels, _ = read_data(data_file)
    scores_and_lens = pickle.load(open(scores_file, "rb"))
    X = featurize(scores_and_lens)
    y = map_labels(labels)
    y = [1 if label > 0 else 0 for label in y]

    lr = LogisticRegression(max_iter=1000, penalty="none", fit_intercept=False)
    lr.fit(X, y)
    print("Coefficients: ", lr.coef_)
    print("Intercept: ", lr.intercept_)
    print("Classification acc: ", lr.score(X, y))


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
