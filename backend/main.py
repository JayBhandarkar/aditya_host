import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Translation API",
    description="Translate Nepali/Sinhala to English using fine-tuned MBART",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "ne_NP"  # Default to Nepali
    tgt_lang: str = "en_XX"  # Default to English

def load_model_lazy():
    global tokenizer, model, device
    
    if tokenizer is None or model is None:
        MODEL_ID = os.getenv("MODEL_ID", "Nikss2709/Mbart-nepali-sinhala-finetuned")
        
        print("Loading tokenizer...")
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_ID)
        
        print("Loading model with low memory usage...")
        model = MBartForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            model = model.to(device)
        model.eval()
        
        print(f"Model loaded on: {device}")

def translate_text(text: str, src_lang: str, tgt_lang: str):
    load_model_lazy()
    
    if not tokenizer or not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
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

@app.get("/")
def home():
    return {"message": "Translation API is running!", "status": "healthy"}

@app.post("/translate")
def translate_api(req: TranslateRequest):
    try:
        result = translate_text(req.text, req.src_lang, req.tgt_lang)
        return {"translated_text": result, "source_language": req.src_lang, "target_language": req.tgt_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/supported-languages")
def get_supported_languages():
    return {
        "source_languages": [
            {"code": "ne_NP", "name": "Nepali", "display": "नेपाली"},
            {"code": "si_LK", "name": "Sinhala", "display": "සිංහල"}
        ],
        "target_languages": [
            {"code": "en_XX", "name": "English", "display": "English"}
        ]
    }

@app.get("/health")
def health_check():
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": device if device else "unknown"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)