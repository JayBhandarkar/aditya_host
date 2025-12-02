import os
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")

print("Loading tokenizer...")
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_ID)

print("Loading model...")
model = MBartForConditionalGeneration.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Model loaded on: {device}")

def translate_text(text: str, src_lang: str, tgt_lang: str):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    generated = model.generate(
        **encoded,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output
