import argparse
import json

from nli_writer import JsonWriter, CsvWriter
from pair_counter import PairWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("--mode", type=str, default="csv")
    parser.add_argument("--pairs_path", type=str, default=None)
    return parser.parse_args()

names = [
    "James", "Olivia", "Liam", "Emma", "Noah", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
]

contexts = [
    ("{a} baked {b} a cake.", "{a} baked {b}.", "{a} baked a cake."),
    ("{a} drew {b} a picture.", "{a} drew {b}.", "{a} drew a picture."),
    ("{a} sang {b} a song.", "{a} sang {b}.", "{a} sang a song."),
    ("{a} read {b} a book.", "{a} read {b}.", "{a} read a book."),
]

args = parse_args()
fh = open(args.save_path, "w")
writer = CsvWriter(fh) if args.mode == "csv" else JsonWriter(fh)
pairs = PairWriter()

for context in contexts:
    for a in names:
        for b in names:
            if a == b:
                continue

            prop_b_obj = context[0].format(a=a, b=b)
            prop_b = context[1].format(a=a, b=b)
            prop_obj = context[2].format(a=a)
            writer.write_pair(
                premise=prop_b_obj,
                entailed=prop_obj,
                not_entailed=prop_b,
                pair_type="dative",
            )

            # All examples are paired here.
            pairs.add_current()
            pairs.increment(2)

fh.close()

if args.pairs_path is not None:
    pairs.save(args.pairs_path)