import argparse
import json

from nli_writer import JsonWriter, CsvWriter
from pair_counter import PairWriter

names = [
    "James", "Olivia", "Liam", "Emma", "Noah", "Ava", "Isabella", "Sophia", "Mia", "Charlotte",
]

contexts = [
    ("{agent} saw {patient}.", "{patient} was seen by {agent}.", "{patient} was seen."),
    ("{agent} taught {patient}.", "{patient} was taught by {agent}.", "{patient} was taught."),
    ("{agent} copied {patient}.", "{patient} was copied by {agent}.", "{patient} was copied."),
    ("{agent} followed {patient}.", "{patient} was followed by {agent}.", "{patient} was followed."),
    ("{agent} trampled {patient}.", "{patient} was trampled by {agent}.", "{patient} was trampled."),
    ("{agent} ridiculed {patient}.", "{patient} was ridiculed by {agent}.", "{patient} was ridiculed."),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("--mode", type=str, default="csv")
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

            prop_active = context[0].format(agent=a, patient=b)
            prop_passive = context[1].format(agent=a, patient=b)
            prop_reduced_a = context[2].format(patient=a)
            prop_reduced_b = context[2].format(patient=b)

            writer.write_pair(
                premise=prop_passive,
                entailed=prop_reduced_b,
                not_entailed=prop_reduced_a,
                pair_type="passive",
            )
            pairs.add_current()
            pairs.increment(2)

            writer.write_pair(
                premise=prop_active,
                entailed=prop_reduced_b,
                not_entailed=prop_reduced_a,
                pair_type="active",
            )
            pairs.add_current()
            pairs.increment(2)

fh.close()

if args.pairs_path is not None:
    pairs.save(args.pairs_path)
