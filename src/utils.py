def read_data(file):
    pairs = []
    labels = []
    ternary = None
    with open(file) as f:
        for line in f:
            p, h, label = line.strip().split("\t")
            pairs.append((p, h))
            labels.append(label)
            if label == "n":
                assert ternary is not False
                ternary = True
            elif label == "nc":
                assert ternary is not True
                ternary = False
    assert ternary is not None
    return pairs, labels, ternary
