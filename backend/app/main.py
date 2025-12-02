from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logical import translate_text
import uvicorn

app = FastAPI(
    title="MBART Translation API",
    description="Translate Nepali/Sinhala to English using fine-tuned MBART",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

@app.get("/")
def home():
    return {"message": "Translation API is running!"}

@app.post("/translate")
def translate_api(req: TranslateRequest):
    result = translate_text(req.text, req.src_lang, req.tgt_lang)
    return {"translated_text": result}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
