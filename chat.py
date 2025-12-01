import gradio as gr
import os
from typing import Optional, Tuple, Any, Dict

# Import model loaders
from model_loader_mt5 import load_model as load_mt5_model, translate as translate_mt5
from model_loader_rnn import load_model as load_rnn_model, Lang, translate as translate_rnn
from model_loader_skipgram import load_model as load_skipgram_model, translate as translate_skipgram

# Configuration
MODEL_DIR = "saved_models"

# Dialect mapping
DIALECT_NAMES = {
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
    "CR": "Costa Rica",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "HN": "Honduras",
    "NI": "Nicaragua",
    "PA": "Panama",
    "PE": "Peru",
    "PR": "Puerto Rico",
    "SV": "El Salvador",
    "UY": "Uruguay",
    "VE": "Venezuela"
}

# Model type mapping
MODEL_TYPES = {
    "MT5": "mt5",
    "RNN": "rnn",
    "Skipgram": "skipgram"
}

# Global variables for model management
current_model: Optional[Any] = None
current_tokenizer: Optional[Any] = None  # For MT5 models
current_model_type: Optional[str] = None
current_dialect: Optional[str] = None
current_input_lang: Optional[Any] = None  # For RNN models
current_output_lang: Optional[Any] = None  # For RNN models


def get_model_filename(model_type: str, dialect_code: str) -> str:
    """Get the model filename based on model type and dialect."""
    if model_type == "mt5":
        return "my_saved_mt5_model"
    elif model_type == "rnn":
        return f"model_es-{dialect_code}.pt" # rnn_
    elif model_type == "skipgram":
        return f"skipgram_model_es-{dialect_code}.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_for_dialect(model_type: str, dialect_code: str) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]:
    """Load a model for a specific model type and dialect.
    
    Returns:
        Tuple of (model, tokenizer, input_lang, output_lang)
        For MT5: (model, tokenizer, None, None)
        For RNN: (model_dict, None, input_lang, output_lang)
        For Skipgram: (model_dict, None, None, None)
    """
    model_filename = get_model_filename(model_type, dialect_code)
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if model_type == "mt5":
        model, tokenizer = load_mt5_model(model_path)
        return model, tokenizer, None, None
    elif model_type == "rnn":
        model_dict, input_lang, output_lang = load_rnn_model(model_path)
        return model_dict, None, input_lang, output_lang
    elif model_type == "skipgram":
        model_dict = load_skipgram_model(model_path)
        return model_dict, None, None, None
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def translate_text(text_en: str, model: Any, model_type: str, 
                   tokenizer: Optional[Any] = None,
                   input_lang: Optional[Any] = None, 
                   output_lang: Optional[Any] = None) -> str:
    """Translate English text to Spanish using the appropriate model."""
    if not text_en or not text_en.strip():
        return ""
    
    try:
        if model_type == "mt5":
            # For MT5, we have separate model and tokenizer
            return translate_mt5(
                text_en=text_en,
                model=model,
                tokenizer=tokenizer,
            )
        elif model_type == "rnn":
            # For RNN, model is a dict with encoder/decoder
            return translate_rnn(model, input_lang, output_lang, text_en)
        elif model_type == "skipgram":
            # For Skipgram, model is a dict
            return translate_skipgram(model, text_en)
        else:
            return f"Unknown model type: {model_type}"
    except Exception as e:
        return f"Translation error: {str(e)}"


def chat_fn(message: str, history: list, dialect: str, model_type: str) -> Tuple[list, str]:
    """Handle chat messages and return responses."""
    global current_model, current_tokenizer, current_model_type, current_dialect, current_input_lang, current_output_lang
    
    # Extract dialect code from selection (format: "Argentina (AR)")
    dialect_code = dialect.split("(")[-1].rstrip(")")
    
    # Extract model type code
    model_type_code = MODEL_TYPES.get(model_type, "mt5")
    
    # Load model if needed or if dialect/model changed
    needs_reload = (
        current_model is None or 
        current_model_type != model_type_code or 
        (current_dialect != dialect_code and model_type_code != "mt5")
    )
    
    if needs_reload:
        try:
            current_model, current_tokenizer, current_input_lang, current_output_lang = load_model_for_dialect(
                model_type_code, dialect_code
            )
            current_model_type = model_type_code
            current_dialect = dialect_code
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"
            history.append((message, error_msg))
            return history, ""
    
    # Translate the message
    try:
        response = translate_text(
            message, 
            current_model, 
            current_model_type,
            current_tokenizer,
            current_input_lang,
            current_output_lang
        )
        history.append((message, response))
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        history.append((message, error_msg))
    
    return history, ""


def reset_conversation(dialect: str, model_type: str) -> list:
    """Reset conversation when dialect or model changes."""
    global current_model, current_tokenizer, current_model_type, current_dialect, current_input_lang, current_output_lang
    current_model = None
    current_tokenizer = None
    current_model_type = None
    current_dialect = None
    current_input_lang = None
    current_output_lang = None
    return []


# Create Gradio interface
with gr.Blocks(title="English to Spanish Dialect Translator") as demo:
    gr.Markdown("# English to Spanish Dialect Translation Chat")
    gr.Markdown("Select a model type and Spanish dialect, then start translating English text!")
    
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_TYPES.keys()),
            label="Select Model Type",
            value= "MT5",
            interactive=True
        )
        dialect_dropdown = gr.Dropdown(
            choices=[f"{name} ({code})" for code, name in DIALECT_NAMES.items()],
            label="Select Spanish Dialect",
            value="Argentina (AR)",
            interactive=True
        )
    
    chatbot = gr.Chatbot(
        label="Translation Chat",
        height=500,
        show_copy_button=True
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Enter English text to translate",
            placeholder="Type your message in English...",
            scale=4,
            container=False
        )
        submit_btn = gr.Button("Translate", scale=1, variant="primary")
    
    # Event handlers
    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, dialect_dropdown, model_dropdown],
        outputs=[chatbot, msg]
    )
    
    submit_btn.click(
        chat_fn,
        inputs=[msg, chatbot, dialect_dropdown, model_dropdown],
        outputs=[chatbot, msg]
    )
    
    # Reset conversation when dialect or model changes
    def reset_on_change(dialect, model_type):
        return reset_conversation(dialect, model_type)
    
    dialect_dropdown.change(
        reset_on_change,
        inputs=[dialect_dropdown, model_dropdown],
        outputs=[chatbot]
    )
    
    model_dropdown.change(
        reset_on_change,
        inputs=[dialect_dropdown, model_dropdown],
        outputs=[chatbot]
    )
    
    gr.Markdown("### Instructions:")
    gr.Markdown("- Select a model type (MT5, RNN, or Skipgram) from the dropdown")
    gr.Markdown("- Select a Spanish dialect from the dropdown")
    gr.Markdown("- Type your English text in the input box and press Enter or click Translate")
    gr.Markdown("- Changing the model type or dialect will reset the conversation")
    gr.Markdown("- The selected model will translate your English text to the selected Spanish dialect")

if __name__ == "__main__":
    demo.launch(share=False)