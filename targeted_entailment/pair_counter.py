import json

class PairWriter:
    def __init__(self, pairs=[], count=0):
        self.pairs = []
        self.count = count

    def increment(self, n=1):
        self.count += n

    def add_current(self):
        self.pairs.append((self.count, self.count + 1))
    
    def add(self, i, j):
        self.pairs.append((i, j))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"pairs": self.pairs}, fh)
            print(f"Saved pairs to {path}")
