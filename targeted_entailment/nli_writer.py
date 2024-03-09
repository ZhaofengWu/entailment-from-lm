"""Utilities for writing synthetic NLI datasets"""

from abc import ABCMeta, abstractmethod
import json

class NliWriter(metaclass=ABCMeta):
    @abstractmethod
    def write_example(self, premise, hypothesis, label, ex_type=None) -> None:
        return NotImplemented

    def write_pair(self, premise, entailed, not_entailed, not_label=1, pair_type=None):
        self.write_example(premise, entailed, 0, ex_type=pair_type)
        self.write_example(premise, not_entailed, not_label, ex_type=pair_type)

class JsonWriter(NliWriter):
    def __init__(self, fh):
        self.fh = fh

    def write_example(self, premise, hypothesis, label, ex_type=None):
        blob = {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
        }
        if ex_type is not None:
            blob["type"] = ex_type
        self.fh.write(json.dumps(blob))
        self.fh.write("\n")

class CsvWriter(NliWriter):
    def __init__(self, fh):
        self.fh = fh

    def write_example(self, premise, hypothesis, label, ex_type=None):
        self.fh.write("\t".join([premise, hypothesis, "e" if label == 0 else "nc"]))
        self.fh.write("\n")
