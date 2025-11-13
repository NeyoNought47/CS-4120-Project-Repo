# pt4
import sentencepiece as spm
from tqdm import tqdm

sp = spm.SentencePieceProcessor()
sp.load("spm/sp16k/spm_bpe16k.model")

splits = ["train", "dev", "test"]

for split in splits:
    for language in ["en", "es"]:
        input_path = f"data/clean/std_es/{split}.{language}"
        output_path = f"data/clean/std_es/{split}.{language}.spm"
        print("Encoding {} to {}".format(input_path, output_path))

        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as out:

            for line in tqdm(fin, desc=f"{split}.{language}"):
                tokens = sp.encode(line.strip(), out_type=str)
                out.write(" ".join(tokens) + "\n")
