"""Model loader for MT5 translation models."""
import torch
import os
from transformers import MT5ForConditionalGeneration, T5TokenizerFast
from typing import Tuple, Optional

# BASE_MODEL_NAME = "google/mt5-small"
TASK_PREFIX = "translate English to Spanish: "
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str) -> Tuple[MT5ForConditionalGeneration, T5TokenizerFast]:
    """Load an MT5 model and tokenizer from a Hugging Face `save_pretrained` directory.

    Args:
        model_path: Path to the directory containing `config.json`, `model.safetensors`,
            `tokenizer_config.json`, etc. (i.e., the directory you passed to `save_pretrained`).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Load model and tokenizer with error checking from the provided directory
    try:
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        return model, tokenizer
    except AttributeError as e:
        print(f"\nAn AttributeError occurred during model/tokenizer loading: {e}")
        print("This often happens if the configuration files (e.g., config.json, tokenizer_config.json) are missing or corrupted in the saved directory.")
        print("Please ensure the model was saved completely and correctly to the specified path, and try re-running the save cell first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during model/tokenizer loading: {e}")

def translate(
    text_en, # The English text to be translated
    model,   # The MT5 model used for translation
    tokenizer, # The tokenizer corresponding to the MT5 model
    num_beams=8, # Number of beams for beam search. 1 means greedy decoding.
    do_sample=False, # Whether to use sampling; False for deterministic decoding (beam search/greedy)
    max_length=128, # Maximum length of the generated target sequence
    length_penalty=1, # Penalty for generating longer sequences
    temperature=1, # Controls randomness in sampling. Lower values make output more deterministic.
    top_p=None, # Top-p (nucleus) sampling parameter
):
    # Prepare the input text with the task prefix
    input_text = TASK_PREFIX + text_en
    # Tokenize the input text and move it to the appropriate device (CPU/GPU)
    inputs = tokenizer(
        input_text,
        return_tensors="pt", # Return PyTorch tensors
        truncation=True,     # Truncate sequences longer than max_source_length
        max_length=MAX_SOURCE_LENGTH,
    ).to(device)

    # Define generation arguments
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "do_sample": do_sample,
        "temperature": temperature,
    }

    # Add top_p to generation arguments if specified
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    # Generate the output sequence (translated text token IDs)
    output_ids = model.generate(**inputs, **gen_kwargs)
    # Decode the generated token IDs back into human-readable text, skipping special tokens
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print(os.listdir("saved_models/my_saved_mt5_model"))
# model, tokenizer = load_model("saved_models/my_saved_mt5_model")
# print(model, tokenizer)
# print("Successful!")
