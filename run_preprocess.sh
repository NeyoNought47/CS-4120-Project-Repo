#!/bin/bash
set -e

echo "Step 1: Cleaning..."
python clean_es_en.py

echo "Step 2: Splitting train/dev/test..."
python split_es_en.py

echo "Step 3: Training SentencePiece..."
python train_sentencepiece.py

echo "Step 4: Encoding train/dev/test with SPM..."
python encode_es_en.py

echo "All steps completed!"

