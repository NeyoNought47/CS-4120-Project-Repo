"""Test script for RNN model loader."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, Any, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

class Lang:
    """Language vocabulary class."""
    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence: str):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    """Encoder RNN for sequence-to-sequence translation."""
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float = 0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    """Attention-based Decoder RNN for sequence-to-sequence translation."""
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.1, max_length: int = MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

def normalizeString(s: str) -> str:
    """Normalize a string for processing."""
    s = s.lower().strip()
    s = re.sub(r"([.!?¿¡,])", r" \1", s)
    s = re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚÑ.!?¿¡,]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def indexesFromSentence(lang: Lang, sentence: str) -> list:
    """Convert sentence to list of word indices."""
    words = sentence.split(' ')
    indexes = []
    for word in words:
        if word and word in lang.word2index:
            indexes.append(lang.word2index[word])
        # Skip unknown words (as in the original notebook)
    return indexes


def tensorFromSentence(lang: Lang, sentence: str) -> torch.Tensor:
    """Convert sentence to tensor."""
    indexes = indexesFromSentence(lang, sentence)
    if not indexes:  # Handle empty sentence
        indexes = [SOS_token]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def translate(model_dict: Dict[str, Any], input_lang: Lang, output_lang: Lang, text_en: str) -> str:
    """Translate English text to Spanish using the RNN model."""
    if not text_en or not text_en.strip():
        return ""
    
    encoder = model_dict["encoder"]
    decoder = model_dict["decoder"]
    
    # Normalize input
    normalized_text = normalizeString(text_en)
    
    if not normalized_text:
        return ""
    
    # Convert to tensor
    try:
        input_tensor = tensorFromSentence(input_lang, normalized_text)
        input_length = input_tensor.size(0)
        
        if input_length == 0:
            return ""
    except KeyError as e:
        return f"Error: Word not in vocabulary: {str(e)}"
    except Exception as e:
        return f"Error processing input: {str(e)}"
    
    encoder.eval()
    decoder.eval()
    
    try:
        with torch.no_grad():
            # Encode
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decode
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            decoded_words = []
            
            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_token_id = topi.item()
                
                if decoder_token_id == EOS_token:
                    break
                else:
                    if decoder_token_id in output_lang.index2word:
                        decoded_words.append(output_lang.index2word[decoder_token_id])
                    else:
                        # Handle unknown index - might be out of vocabulary
                        break
                
                # Prepare next decoder input - maintain shape [1, 1]
                decoder_input = topi.detach()

        result = " ".join(decoded_words)
        return result if result else "[No translation generated]"
    except Exception as e:
        import traceback
        return f"Translation error: {str(e)}\n{traceback.format_exc()}"

def load_model(model_path: str) -> Tuple[Dict[str, Any], Lang, Lang]:
    """Load a model from a file."""
    # Step 1: Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Step 2: Extract components
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']
    hidden_size = checkpoint.get('hidden_size', 256)

    # Step 3: Create model instances
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # Step 4: Load state dicts
    encoder.load_state_dict(encoder_state_dict) # , strict=False
    decoder.load_state_dict(decoder_state_dict) # , strict=False

    # Step 5: Create model dict
    model_dict = {
            "encoder": encoder,
            "decoder": decoder,
            "hidden_size": hidden_size
    }
    
    return model_dict, input_lang, output_lang

# model_dict, input_lang, output_lang = load_model("saved_models/model_es-AR.pt")
# text = "Hello, how are you?"
# translated = translate(model_dict, input_lang, output_lang, text)
# print(translated)
