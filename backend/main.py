import os
import torch
import requests
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from dotenv import load_dotenv
import uvicorn
import gc

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
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://host-one-psi.vercel.app"
    ],
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
        
        print("Loading model with memory optimization...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = MBartForConditionalGeneration.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        model = model.to(device)
        model.eval()
        
        # Memory optimization without changing model
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        
        # Compile model for memory efficiency (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Model loaded on: {device} with memory optimizations")

def translate_text(text: str, src_lang: str, tgt_lang: str):
    try:
        print(f"\n=== Starting translation ===")
        print(f"Text: '{text}'")
        print(f"From {src_lang} to {tgt_lang}")
        
        load_model_lazy()
        
        if not tokenizer or not model:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        print(f"Setting tokenizer languages...")
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        print(f"Encoding text...")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        print(f"Text encoded, shape: {encoded['input_ids'].shape}")

        print(f"Generating translation...")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=128,
                num_beams=2,
                early_stopping=True,
                use_cache=False
            )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Translation generated")

        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Decoded output: '{output}'")
        print(f"=== Translation complete ===\n")
        return output
    except Exception as e:
        print(f"\n!!! ERROR in translate_text: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
def home():
    return {"message": "Translation API is running!", "status": "healthy"}

@app.post("/translate")
def translate_api(req: TranslateRequest):
    try:
        print(f"\nReceived translation request: {req}")
        result = translate_text(req.text, req.src_lang, req.tgt_lang)
        return {"translated_text": result, "source_language": req.src_lang, "target_language": req.tgt_lang}
    except Exception as e:
        print(f"\n!!! ERROR in translate_api: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/supported-languages")
def get_supported_languages():
    return {
        "languages": [
            {"code": "ne_NP", "name": "Nepali", "display": "नेपाली"},
            {"code": "si_LK", "name": "Sinhala", "display": "සිංහල"},
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

@app.post("/ocr/printed")
async def extract_printed_text(file: UploadFile = File(...)):
    image_bytes = await file.read()
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    
    url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {'requests': [{'image': {'content': encoded_image}, 'features': [{'type': 'TEXT_DETECTION'}]}]}
    headers = {'Referer': 'http://localhost:3000', 'Origin': 'http://localhost:3000'}
    response = requests.post(url, json=payload, headers=headers)
    
    result = response.json()
    if 'responses' in result and result['responses']:
        resp = result['responses'][0]
        annotations = resp.get('textAnnotations', [])
        if annotations:
            return {"extracted_text": annotations[0]['description'], "type": "printed"}
    
    return {"extracted_text": "No text found in image", "type": "printed"}

@app.post("/ocr/handwritten")
async def extract_handwritten_text(file: UploadFile = File(...)):
    image_bytes = await file.read()
    api_key = os.getenv('GOOGLE_VISION_API_KEY')
    
    url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {'requests': [{'image': {'content': encoded_image}, 'features': [{'type': 'DOCUMENT_TEXT_DETECTION'}]}]}
    headers = {'Referer': 'http://localhost:3000', 'Origin': 'http://localhost:3000'}
    response = requests.post(url, json=payload, headers=headers)
    
    result = response.json()
    if 'responses' in result and result['responses']:
        resp = result['responses'][0]
        full_text = resp.get('fullTextAnnotation', {})
        if full_text and 'text' in full_text:
            return {"extracted_text": full_text['text'], "type": "handwritten"}
        annotations = resp.get('textAnnotations', [])
        if annotations:
            return {"extracted_text": annotations[0]['description'], "type": "handwritten"}
    
    return {"extracted_text": "No text found in image", "type": "handwritten"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)