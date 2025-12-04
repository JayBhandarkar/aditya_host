'use client'

import { useState } from 'react'
import Layout from '@/components/Layout'
import { createWorker, PSM } from 'tesseract.js'

export default function ImageUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [extractedText, setExtractedText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [loading, setLoading] = useState(false)
  const [extracting, setExtracting] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [ocrProgress, setOcrProgress] = useState(0)

  const handleFileSelect = (file: File) => {
    if (file && (file.type.startsWith('image/') || file.type === 'application/pdf')) {
      setSelectedFile(file)
      extractTextFromFile(file)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }

  const preprocessImage = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')!
      const img = new Image()
      
      img.onload = () => {
        canvas.width = img.width * 2
        canvas.height = img.height * 2
        
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const data = imageData.data
        
        for (let i = 0; i < data.length; i += 4) {
          const avg = (data[i] + data[i + 1] + data[i + 2]) / 3
          const threshold = avg > 128 ? 255 : 0
          data[i] = data[i + 1] = data[i + 2] = threshold
        }
        
        ctx.putImageData(imageData, 0, 0)
        resolve(canvas.toDataURL())
      }
      
      img.src = URL.createObjectURL(file)
    })
  }

  const handleOCRExtraction = async (file: File, type: 'printed' | 'handwritten') => {
    setSelectedFile(file)
    setExtracting(true)
    setOcrProgress(0)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const endpoint = type === 'printed' ? '/ocr/printed' : '/ocr/handwritten'
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData
      })
      
      const data = await response.json()
      setExtractedText(data.extracted_text || 'No text detected')
    } catch (error) {
      console.error('OCR extraction failed:', error)
      setExtractedText('OCR extraction failed. Please try again.')
    } finally {
      setExtracting(false)
      setOcrProgress(0)
    }
  }

  const extractTextFromFile = async (file: File) => {
    setExtracting(true)
    setOcrProgress(0)
    try {
      const processedImage = await preprocessImage(file)
      
      const worker = await createWorker('nep+sin+eng', 1, {
        logger: m => {
          if (m.status === 'recognizing text') {
            setOcrProgress(Math.round(m.progress * 100))
          }
        }
      })
      
      await worker.setParameters({
        tessedit_pageseg_mode: PSM.SINGLE_BLOCK
      })
      
      const { data: { text } } = await worker.recognize(processedImage)
      await worker.terminate()
      
      const cleanedText = text
        .replace(/[^\u0900-\u097F\u0D80-\u0DFF\u0020-\u007E\s]/g, '')
        .replace(/\s+/g, ' ')
        .trim()
      
      setExtractedText(cleanedText || 'No text detected. Please try a clearer image.')
    } catch (error) {
      console.error('OCR extraction failed:', error)
      setExtractedText('OCR extraction failed. Please try again with a clearer image.')
    } finally {
      setExtracting(false)
      setOcrProgress(0)
    }
  }

  const handleTranslate = async () => {
    if (!extractedText.trim()) return

    setLoading(true)
    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: extractedText,
          src_lang: 'ne_NP',
          tgt_lang: 'en_XX'
        })
      })
      
      const data = await response.json()
      setTranslatedText(data.translated_text)
    } catch (error) {
      console.error('Translation failed:', error)
      setTranslatedText('Translation failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const playAudio = (text: string) => {
    if ('speechSynthesis' in window && text) {
      const utterance = new SpeechSynthesisUtterance(text)
      speechSynthesis.speak(utterance)
    }
  }

  return (
    <Layout>
      <div className="translator-shrine">
        <div className="shrine-header">
          <h1 className="shrine-title">OCR + TRANSLATION</h1>
          <div className="decorative-border"></div>
        </div>

        {/* Upload Area */}
        <div 
          className={`sacred-upload ${dragActive ? 'drag-active' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
        >
          <div className="upload-icon">📄</div>
          <div className="upload-text">Upload Image or PDF</div>
          <p className="upload-subtitle">Extract text from images and PDFs - Nepali & Sinhala Support</p>
          
          <div className="ocr-buttons">
            <button
              onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'
                input.accept = 'image/*'
                input.onchange = (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (file) handleOCRExtraction(file, 'printed')
                }
                input.click()
              }}
              className="ocr-btn printed"
              disabled={extracting}
            >
              📄 Extract Printed Text
            </button>
            
            <button
              onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'
                input.accept = 'image/*'
                input.onchange = (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (file) handleOCRExtraction(file, 'handwritten')
                }
                input.click()
              }}
              className="ocr-btn handwritten"
              disabled={extracting}
            >
              ✍️ Extract Handwritten Text
            </button>
          </div>
          
          <div className="upload-ornament"></div>
          
          {selectedFile && (
            <div className="file-info">
              <span>📎 {selectedFile.name}</span>
            </div>
          )}
        </div>

        {/* Extracted Text Section */}
        <div className="extraction-section">
          <div className="chamber-header">
            <h3 className="chamber-label">Extracted Text</h3>
            <button
              onClick={() => playAudio(extractedText)}
              className="voice-btn"
            >
              <span>🔊</span>
            </button>
          </div>
          
          <textarea
            value={extractedText}
            onChange={(e) => setExtractedText(e.target.value)}
            placeholder={extracting ? `Extracting text... ${ocrProgress}%` : "Extracted text will appear here..."}
            className="sacred-textarea"
            disabled={extracting}
          />
          {extracting && (
            <div className="ocr-progress">
              <div className="progress-bar">
                <div className="progress-fill" style={{width: `${ocrProgress}%`}}></div>
              </div>
              <span className="progress-text">{ocrProgress}% Complete</span>
            </div>
          )}
        </div>

        {/* Translate Button */}
        <div className="transformation-bridge">
          <button
            onClick={handleTranslate}
            disabled={loading || !extractedText.trim()}
            className="transform-btn"
          >
            {loading ? 'TRANSLATING...' : 'TRANSLATE EXTRACTED TEXT'}
          </button>
        </div>

        {/* Translation Result */}
        {translatedText && (
          <div className="target-chamber">
            <div className="chamber-header">
              <h3 className="chamber-label">English Translation</h3>
              <button
                onClick={() => playAudio(translatedText)}
                className="voice-btn"
              >
                <span>🔊</span>
              </button>
            </div>
            
            <textarea
              value={translatedText}
              readOnly
              className="sacred-textarea"
            />
          </div>
        )}
      </div>
    </Layout>
  )
}