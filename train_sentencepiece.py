# pt3
import os
from pathlib import Path
import sentencepiece as spm

train_en = "data/clean/std_es/train.en"
train_es = "data/clean/std_es/train.es"

os.makedirs("spm/sp16k", exist_ok=True)

spm.SentencePieceTrainer.Train(
    input=",".join([train_en, train_es]),
    model_prefix="spm/sp16k/spm_bpe16k",
    vocab_size=20000,
    model_type="bpe",
    character_coverage=1.0,
    input_sentence_size=1000000,
    shuffle_input_sentence=True
)
