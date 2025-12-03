'use client'

import { useState, useRef } from 'react'
import axios from 'axios'
import { useLanguage } from '../contexts/LanguageContext'
import WelcomeHeader from './WelcomeHeader'

interface TranslationResponse {
  translated_text: string
  source_lang: string
  target_lang: string
}

export default function TranslationInterface() {
  const { t } = useLanguage()
  const [inputText, setInputText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [sourceLang, setSourceLang] = useState('ne_NP')
  const [targetLang, setTargetLang] = useState('en_XX')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [ocrLoading, setOcrLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const recognitionRef = useRef<any>(null)

  const handleOCR = async (type: 'printed' | 'handwritten') => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/*'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      setOcrLoading(true)
      const formData = new FormData()
      formData.append('file', file)

      try {
        const endpoint = type === 'printed' ? '/ocr/printed' : '/ocr/handwritten'
        const response = await axios.post(`http://localhost:8000${endpoint}`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        setInputText(response.data.extracted_text)
      } catch (err) {
        console.error('OCR failed:', err)
        setError('Text extraction failed')
      } finally {
        setOcrLoading(false)
      }
    }
    input.click()
  }

  const startRecording = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setError('Speech recognition not supported. Use Chrome or Edge.')
      return
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    recognitionRef.current = new SpeechRecognition()
    recognitionRef.current.continuous = true
    recognitionRef.current.interimResults = true
    recognitionRef.current.lang = sourceLang === 'ne_NP' ? 'ne-NP' : sourceLang === 'si_LK' ? 'si-LK' : 'en-XX'

    recognitionRef.current.onresult = (event: any) => {
      let finalTranscript = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript + ' '
        }
      }
      if (finalTranscript) {
        setInputText(prev => prev + finalTranscript)
      }
    }

    recognitionRef.current.onerror = () => setIsRecording(false)
    recognitionRef.current.onend = () => setIsRecording(false)
    recognitionRef.current.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }
  }

  const speakText = () => {
    if (!translatedText.trim()) return
    
    const utterance = new SpeechSynthesisUtterance(translatedText)
    utterance.lang = targetLang === 'ne_NP' ? 'ne-NP' : targetLang === 'si_LK' ? 'si-LK' : 'en-US'
    window.speechSynthesis.speak(utterance)
  }

  const downloadAsWord = () => {
    if (!translatedText.trim()) return

    const content = `Original Text:\n${inputText}\n\nTranslated Text:\n${translatedText}`
    const blob = new Blob([content], { type: 'application/msword' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'translation.doc'
    link.click()
    URL.revokeObjectURL(url)
  }

  const handleTranslate = async () => {
    if (!inputText.trim()) return

    setLoading(true)
    setError('')
    
    try {
      const response = await axios.post<TranslationResponse>('http://localhost:8000/translate', {
        text: inputText,
        src_lang: sourceLang,
        tgt_lang: targetLang
      })
      
      console.log('Translation response:', response.data)
      const translation = response.data.translated_text || 'No translation received'
      setTranslatedText(translation)
      
      // Show success message
      if (translation && translation !== 'No translation received') {
        setTimeout(() => {
          const outputElement = document.querySelector('textarea[readonly]')
          if (outputElement) {
            outputElement.style.backgroundColor = '#e8f5e8'
            setTimeout(() => {
              outputElement.style.backgroundColor = '#f0f8ff'
            }, 1000)
          }
        }, 100)
      }
    } catch (err: any) {
      console.error('Translation error:', err)
      const errorMessage = err.response?.data?.detail || err.message || 'Translation failed. Please try again.'
      setError(errorMessage)
      setTranslatedText('Translation failed - please try again')
    } finally {
      setLoading(false)
    }
  }



  return (
    <>
      <div className="translator-shrine">
      <div className="shrine-header">
        <h1 className="shrine-title">{t('translate.title').toUpperCase()}</h1>
        <div className="decorative-border"></div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="source-chamber">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ color: '#8b4513', fontSize: '1.1rem', fontWeight: '600', margin: 0 }}>From</h3>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <select
                value={sourceLang}
                onChange={(e) => setSourceLang(e.target.value)}
                className="cultural-select"
              >
                <option value="ne_NP">{t('common.nepali')}</option>
                <option value="si_LK">{t('common.sinhala')}</option>
                <option value="en_XX">{t('common.english')}</option>
              </select>
              <button
                onClick={isRecording ? stopRecording : startRecording}
                style={{ 
                  padding: '8px 12px', 
                  background: isRecording ? '#dc3545' : '#007bff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                {isRecording ? '⏹️' : '🎤'}
              </button>
            </div>
          </div>
          
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={t('translate.placeholder')}
            className="sacred-textarea"
          />
        </div>

        <div style={{ textAlign: 'center' }}>
          <button
            onClick={handleTranslate}
            disabled={loading || !inputText.trim()}
            className="transform-btn"
          >
            {loading ? t('translate.loading').toUpperCase() : t('translate.button').toUpperCase()}
          </button>
        </div>

        <div className="target-chamber">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ color: '#8b4513', fontSize: '1.1rem', fontWeight: '600', margin: 0 }}>To</h3>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <select 
                value={targetLang}
                onChange={(e) => setTargetLang(e.target.value)}
                className="cultural-select"
              >
                <option value="en_XX">{t('common.english')}</option>
                <option value="ne_NP">{t('common.nepali')}</option>
                <option value="si_LK">{t('common.sinhala')}</option>
              </select>
              <button
                onClick={speakText}
                disabled={!translatedText}
                style={{ 
                  padding: '8px 12px', 
                  background: translatedText ? '#28a745' : '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: translatedText ? 'pointer' : 'not-allowed'
                }}
              >
                🔊
              </button>
              <button
                onClick={downloadAsWord}
                disabled={!translatedText}
                style={{ 
                  padding: '8px 12px', 
                  background: translatedText ? '#17a2b8' : '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: translatedText ? 'pointer' : 'not-allowed'
                }}
              >
                📥
              </button>
            </div>
          </div>
          
          <textarea
            value={translatedText}
            readOnly
            placeholder="Translation will appear here..."
            className="sacred-textarea"
            style={{ 
              backgroundColor: '#f0f8ff', 
              minHeight: '200px',
              border: '3px solid #daa520',
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#2F1B14'
            }}
          />
        </div>
      </div>

      {error && (
        <div className="mt-6 p-4 bg-red-100 border-2 border-red-300 text-red-700 rounded-xl shadow-md">
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
    </>
  )
}