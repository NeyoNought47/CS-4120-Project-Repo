import dill as pickle
from nltk.translate import AlignedSent, IBMModel1, bleu_score
from nltk.tokenize import word_tokenize
from collections import defaultdict


def load_model(model_path):
    """Loads a trained model from disk using pickle.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Model object with translation_table attribute (compatible with translate/evaluate functions)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def build_translation_dict(model):
    """Builds a translation dictionary from the trained model.
    
    Args:
        model: Trained IBMModel1 instance
    
    Returns:
        Dictionary mapping source words to target words
    """
    translation_dict = {}
    s_to_t_probs = defaultdict(list)
    
    # Extract translation probabilities
    for t in model.translation_table:
        for s in model.translation_table[t]:
            prob = model.translation_table[t][s]
            if prob > 1e-6:
                s_to_t_probs[s].append((t, prob))
    
    # For each source word, pick the target word with highest probability
    for s in s_to_t_probs:
        best_t = sorted(s_to_t_probs[s], key=lambda x: x[1], reverse=True)[0][0]
        translation_dict[s] = best_t
    
    return translation_dict

def translate(model, text_en):
    """Translates source tokens using the IBMModel1 model."""
    translation_dict = build_translation_dict(model)
    translated = []
    
    for word in text_en.lower().split():
        if word in translation_dict:
            translated.append(translation_dict[word])
        else:
            translated.append(word)  # Keep original if no translation found
    
    return " ".join(translated)


# model = load_model("models/es-AR_ibm1.pkl")
# print("Model loaded!")
# print(translate(model, "They follow your mouse"))
