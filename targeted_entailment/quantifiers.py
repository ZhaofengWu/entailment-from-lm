"""Generate paired data for quantifiers/numbers experiments."""

import argparse
import json

from nli_writer import JsonWriter, CsvWriter
from pair_counter import PairWriter

names = [
    "James", "Olivia", "Liam", "Emma", "Noah", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
]

contexts = [
    "{det} of the kids played with {ent}.",
    "{det} of the dogs barked at {ent}.",
    "{det} of the skis were bent by {ent}.",
    "{det} of the stones are older than {ent}.",
    "{det} of the doors are open for {ent}.",
    "{det} of {ent}'s neighbors are friendly.",
    "{det} of {ent}'s friends are tall.",
    "{det} of their houses are bigger than {ent}'s.",
    "{det} of {ent}'s crops failed.",
    "{det} of {ent}'s animals escaped.",
    "{det} of them cried with {ent}.",
    "{det} of the teams played against {ent}.",
    "{det} of us were invited by {ent}.",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("--mode", type=str, default="csv")
    parser.add_argument("--type", type=str, choices=["quantifiers", "numbers"], default="quantifiers")
    parser.add_argument("--allternatives", nargs="+", default=[],
                        help="Alternatives to all")
    parser.add_argument("--pairs_path", type=str, default=None)
    return parser.parse_args()

def write_quantifiers(writer, context, name):
    """Generate paired examples with all, some, none.
      * "all" is always the premise and others are hypotheses.
      * Add alternatives to "all" specified by args.allternatives.
    """
    prop_some = context.format(det="Some", ent=name)
    prop_none = context.format(det="None", ent=name)

    for all_ in ["all"] + args.allternatives:
        prop_all = context.format(det=all_.title(), ent=name)
        writer.write_pair(
            premise=prop_all,
            entailed=prop_some,
            not_entailed=prop_none,
            not_label=2,  # contradiction
            pair_type="all",
        )
        pairs.add_current()
        pairs.increment(2)

def write_numbers(writer, context, name):
    prop_1 = context.format(det="At least one", ent=name).replace("are", "is")
    prop_2 = context.format(det="At least two", ent=name)
    prop_3 = context.format(det="At least three", ent=name)

    writer.write_pair(
        premise=prop_2,
        entailed=prop_1,
        not_entailed=prop_3,
    )
    pairs.add_current()
    pairs.increment(2)

if __name__ == "__main__":
    args = parse_args()
    fh = open(args.save_path, "w")
    writer = CsvWriter(fh) if args.mode == "csv" else JsonWriter(fh)
    pairs = PairWriter()

    for context in contexts:
        for name in names:
            if args.type == "quantifiers":
                write_quantifiers(writer, context, name)
            else:
                write_numbers(writer, context, name)
    fh.close()

    if args.pairs_path is not None:
        pairs.save(args.pairs_path)
