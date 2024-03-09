import argparse
import json

from nli_writer import JsonWriter, CsvWriter
from pair_counter import PairWriter

names = [
    "James", "Olivia", "Liam", "Emma", "Noah", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
]

contexts = [
    ("{sing} is going to Canada.", "{plur} are going to Canada."),
    ("{sing} plays guitar.", "{plur} play guitar."),
    ("I saw {sing}.", "I saw {plur}."),
    ("He likes {sing}.", "He likes {plur}."),
    ("{sing} is hungry.", "{plur} are hungry."),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("--mode", type=str, default="csv")
    parser.add_argument("--paired_only", action="store_true",
                        help="Only write premises with entailed/non-entailed hypotheses")
    parser.add_argument("--pairs_path", type=str, default=None)
    return parser.parse_args()

args = parse_args()
fh = open(args.save_path, "w")
writer = CsvWriter(fh) if args.mode == "csv" else JsonWriter(fh)
pairs = PairWriter()

for context in contexts:
    for a in names:
        for b in names:
            if a == b:
                continue

            prop_a = context[0].format(sing=a)
            prop_b = context[0].format(sing=b)
            prop_a_and_b = context[1].format(plur=f"{a} and {b}")
            prop_a_or_b = context[0].format(sing=f"{a} or {b}")

            # Labels: 0 (entailment), 1 (neutral), 2 (contradiction)
            writer.write_pair(
                premise=prop_a,
                entailed=prop_a_or_b,
                not_entailed=prop_a_and_b,
                pair_type="a",
            )
            pairs.add_current()
            pairs.increment(2)

            writer.write_pair(
                premise=prop_b,
                entailed=prop_a_or_b,
                not_entailed=prop_a_and_b,
                pair_type="a",
            )
            pairs.add_current()
            pairs.increment(2)

            # With and/or as premise, there is no pairing since labels are the same.
            # Could potentially add pair: A or B, A, A and B
            if args.paired_only:
                continue

            # A and B does entail atomic premise A.
            writer.write_example(
                premise=prop_a_and_b,
                hypothesis=prop_a,
                label=0,
                ex_type="A and B => A",
            )
            writer.write_example(
                premise=prop_a_and_b,
                hypothesis=prop_b,
                label=0,
                ex_type="A and B => B",
            )
            pairs.increment(2)

            # A or B does not entail atomic premise A.
            writer.write_example(
                premise=prop_a_or_b,
                hypothesis=prop_a,
                label=1,
                ex_type="A or B !=> A",
            )
            writer.write_example(
                premise=prop_a_or_b,
                hypothesis=prop_b,
                label=1,
                ex_type="A or B !=> B",
            )
            pairs.increment(2)

fh.close()

if args.pairs_path is not None:
    pairs.save(args.pairs_path)
