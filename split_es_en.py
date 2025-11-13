# pt2
from pathlib import Path
import random

random.seed(66)

data_path = Path("data/clean/std_es")
en_lines = (data_path / "all.en").read_text(encoding="utf-8").splitlines()
es_lines = (data_path / "all.es").read_text(encoding="utf-8").splitlines()

if len(en_lines) != len(es_lines):
    raise ValueError("EN/ES line counts must match.")

pairs = list(zip(en_lines, es_lines))
length = len(pairs)
index = list(range(length))
random.shuffle(index)

n_test = max(5000, int(length * 0.05))
n_dev = max(5000, int(length * 0.05))
n_train = length - n_test - n_dev

splits = {
    "train": index[:n_train],
    "dev":   index[n_train:n_train + n_dev],
    "test":  index[n_train + n_dev:],
}

for name, id_list in splits.items():
    with open(data_path / f"{name}.en", "w", encoding="utf-8") as english, \
         open(data_path / f"{name}.es", "w", encoding="utf-8") as spanish:
        
        for i in id_list:
            english.write(pairs[i][0] + "\n")
            spanish.write(pairs[i][1] + "\n")

print({k: len(v) for k, v in splits.items()})
