'use client'

import { useState, useRef } from 'react'
import Layout from '@/components/Layout'
import WelcomeHeader from '@/components/WelcomeHeader'
import axios from 'axios'

export default function SpeechToText() {
  const [isRecording, setIsRecording] = useState(false)
  const [transcribedText, setTranscribedText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [sourceLang, setSourceLang] = useState('ne-NP')
  const [targetLang, setTargetLang] = useState('en_XX')
  const [isTranslating, setIsTranslating] = useState(false)
  const [error, setError] = useState('')
  const recognitionRef = useRef<any>(null)

  const startRecording = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setError('Speech recognition is not supported in your browser. Please use Chrome or Edge.')
      return
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    recognitionRef.current = new SpeechRecognition()
    recognitionRef.current.continuous = true
    recognitionRef.current.interimResults = true
    recognitionRef.current.lang = sourceLang

    recognitionRef.current.onresult = (event: any) => {
      let finalTranscript = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' '
        }
      }
      if (finalTranscript) {
        setTranscribedText(prev => prev + finalTranscript)
      }
    }

    recognitionRef.current.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error)
      setIsRecording(false)
    }

    recognitionRef.current.onend = () => {
      setIsRecording(false)
    }

    recognitionRef.current.start()
    setIsRecording(true)
  }

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleTranslate = async () => {
    if (!transcribedText.trim()) return

    setIsTranslating(true)
    setError('')
    try {
      const langCode = sourceLang === 'ne-NP' ? 'ne_NP' : sourceLang === 'si-LK' ? 'si_LK' : 'en_XX'
      const response = await axios.post('http://localhost:8000/translate', {
        text: transcribedText,
        src_lang: langCode,
        tgt_lang: targetLang
      })
      setTranslatedText(response.data.translated_text)
    } catch (err: any) {
      console.error('Translation error:', err)
      setError(err.response?.data?.detail || 'Translation failed. Please try again.')
    } finally {
      setIsTranslating(false)
    }
  }

  const clearAll = () => {
    setTranscribedText('')
    setTranslatedText('')
    setError('')
  }

  const swapLanguages = () => {
    const temp = sourceLang
    setSourceLang(targetLang === 'en_XX' ? 'en-XX' : targetLang === 'ne_NP' ? 'ne-NP' : 'si-LK')
    setTargetLang(temp === 'ne-NP' ? 'ne_NP' : temp === 'si-LK' ? 'si_LK' : 'en_XX')
    setTranscribedText('')
    setTranslatedText('')
  }

  return (
    <Layout>
      <WelcomeHeader />
      <div className="translator-shrine">
        <div className="shrine-header">
          <h1 className="shrine-title">🎤 SPEECH TO TEXT TRANSLATION</h1>
          <div className="decorative-border"></div>
        </div>

        <div className="translation-chambers">
          <div className="source-chamber">
            <div className="chamber-header">
              <h3 className="chamber-label">SPEAK IN</h3>
              <div className="language-controls">
                <select
                  value={sourceLang}
                  onChange={(e) => setSourceLang(e.target.value)}
                  disabled={isRecording}
                  className="cultural-select"
                >
                  <option value="ne-NP">नेपाली (Nepali)</option>
                  <option value="si-LK">සිංහල (Sinhala)</option>
                  <option value="en-XX">English</option>
                </select>
              </div>
            </div>

            <div style={{ marginBottom: '15px', display: 'flex', gap: '10px' }}>
              {!isRecording ? (
                <button 
                  onClick={startRecording}
                  className="transform-btn"
                  style={{ flex: 1 }}
                >
                  🎤 START RECORDING
                </button>
              ) : (
                <button 
                  onClick={stopRecording}
                  className="transform-btn"
                  style={{ flex: 1, background: '#dc3545' }}
                >
                  ⏹️ STOP RECORDING
                </button>
              )}
              {transcribedText && (
                <button 
                  onClick={clearAll}
                  className="transform-btn"
                  style={{ background: '#6c757d' }}
                >
                  CLEAR
                </button>
              )}
            </div>

            {isRecording && (
              <div style={{ color: '#dc3545', marginBottom: '10px', fontWeight: 'bold', textAlign: 'center' }}>
                🔴 Recording in progress...
              </div>
            )}
            
            <textarea
              value={transcribedText}
              onChange={(e) => setTranscribedText(e.target.value)}
              placeholder="Your speech will appear here..."
              className="sacred-textarea"
            />
          </div>

          <div className="transformation-bridge">
            <button
              onClick={swapLanguages}
              className="swap-btn"
              style={{ marginBottom: '10px' }}
            >
              ⇄
            </button>
            <button
              onClick={handleTranslate}
              disabled={isTranslating || !transcribedText.trim()}
              className="transform-btn"
            >
              {isTranslating ? 'TRANSLATING...' : 'TRANSLATE'}
            </button>
          </div>

          <div className="target-chamber">
            <div className="chamber-header">
              <h3 className="chamber-label">TRANSLATE TO</h3>
              <div className="language-controls">
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="cultural-select"
                >
                  <option value="en_XX">English</option>
                  <option value="ne_NP">नेपाली (Nepali)</option>
                  <option value="si_LK">සිංහල (Sinhala)</option>
                </select>
              </div>
            </div>
            
            <textarea
              value={translatedText}
              readOnly
              placeholder="Translation will appear here..."
              className="sacred-textarea"
              style={{ 
                backgroundColor: '#f0f8ff',
                border: '3px solid #daa520',
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
    </Layout>
  )
}
