# CS-4120-Project-Repo
This project is split into mainly 3 parts: data cleaning, model training, and inference. 

### Data Cleaning:
- Make sure the raw data is in place:
  - OPUS GNOME dialect files under `data/dialect/<dialect>/...`
  - Standard Spanish OPUS data under `en-es.txt/`
  - Extra Tatoeba data merged into `spa.en` / `spa.es`

- In the project root, run:
  `run_preprocess.sh`

  Or run separately:
  - `clean_es_en.py` → `split_es_en.py` → `train_sentencepiece.py` → `encode_es_en.py`

###  Training: 
- For SMT: run `smt_baseline.ipynb`
- For RNN: run `rnn.ipynb`
- For mT5: run `mt5_finetune.ipynb`
  - Save trained models to `models` directory

###  Inference:
- Run `chat.py`
  - Load from `models` directory
