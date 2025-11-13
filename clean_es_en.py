# pt1
import unicodedata, re
from pathlib import Path

dialect = "es-VE"
dialect_name = dialect.split("-")[1]

folder = Path(f"data/dialect/{dialect}/en-es_{dialect_name}.txt")
en_in = folder / f"GNOME.en-es_{dialect_name}.en"
es_in = folder / f"GNOME.en-es_{dialect_name}.es_{dialect_name}"

out = Path(f"data/clean/{dialect}")
out.mkdir(parents=True, exist_ok=True)

en_all = out / "all.en"
es_all = out / "all.es"

regex = re.compile(r"<[^>]+>")

def normalize(s:str)->str:
    s = unicodedata.normalize("NFC", s).replace("\u00A0"," ")
    s = regex.sub("", s)
    return re.sub(r"\s+"," ", s).strip()

def good_len(en_tokens, es_tokens, mn=1, mx=200, rlo=0.5, rhi=2.0):
    if not (mn <= len(en_tokens) <= mx and mn <= len(es_tokens) <= mx): 
        return False
    r = (len(en_tokens)+1e-9)/(len(es_tokens)+1e-9)
    return rlo <= r <= rhi

seen = set()
kept = 0

with open(en_in, encoding="utf-8") as english, open(es_in, encoding="utf-8") as spanish, \
     open(en_all, "w", encoding="utf-8") as out_english, open(es_all, "w", encoding="utf-8") as out_spanish:
    for en, es in zip(english, spanish):
        en = normalize(en)
        es = normalize(es)
        if not en or not es: 
            continue
        if not good_len(en.split(), es.split()): 
            continue
        pair=(en, es)
        if pair in seen: 
            continue
        seen.add(pair)
        kept+=1
        out_english.write(en + "\n")
        out_spanish.write(es + "\n")

print(f"Keeping {kept} pairs.")
