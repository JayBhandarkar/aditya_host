import os
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

# Define model paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BACKEND_DIR, "model")
MERGED_MODEL_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "merged_model")
FINETUNED_MODEL = os.getenv("MODEL_ID", "Nikss2709/Mbart-nepali-sinhala-finetuned")
FALLBACK_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Determine which model to use
if os.path.exists(MERGED_MODEL_DIR) and os.path.exists(os.path.join(MERGED_MODEL_DIR, "config.json")):
    MODEL_PATH = MERGED_MODEL_DIR
    print(f"Using merged model: {MODEL_PATH}")
elif os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
    MODEL_PATH = MODEL_DIR
    print(f"Using adapter model: {MODEL_PATH}")
else:
    MODEL_PATH = FINETUNED_MODEL
    print(f"Using fine-tuned model: {MODEL_PATH}")

print("Loading tokenizer...")
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_PATH)

print("Loading model...")
try:
    model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Trying fallback model: {FALLBACK_MODEL}")
    MODEL_PATH = FALLBACK_MODEL
    model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Model loaded on: {device}")

def translate_text(text: str, src_lang: str, tgt_lang: str):
    try:
        print(f"Translating: '{text}' from {src_lang} to {tgt_lang}")
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1028
        ).to(device)
        print("Text encoded successfully")

        generated = model.generate(
            **encoded,
            max_length=1028,
            num_beams=4,
            early_stopping=True
        )
        print("Translation generated successfully")

        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Decoded output: {output}")
        return output
    except Exception as e:
        print(f"Error in translate_text: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
