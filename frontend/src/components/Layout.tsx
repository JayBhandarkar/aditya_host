'use client'

import { useState } from 'react'
import Sidebar from './Sidebar'
import { useLanguage } from '../contexts/LanguageContext'

export default function Layout({ children }: { children: React.ReactNode }) {
  const { t } = useLanguage()
  const [theme, setTheme] = useState('light')
  const [uiTheme, setUiTheme] = useState('nepali-theme')
  const [language, setLanguage] = useState('en')
  
  return (
    <div className={`cultural-app ${theme} ${uiTheme}`} suppressHydrationWarning>
      <header className="cultural-header">
        <div className="header-ornament"></div>
        <div className="header-content">
          <div className="header-left">
            <h1 className="main-title">संस्कृति</h1>
            <p className="subtitle">{t('translate.title')}</p>
          </div>
          <div className="header-right">
            <select value={language} onChange={(e) => setLanguage(e.target.value)} className="cultural-select">
              <option value="en">English</option>
              <option value="ne">नेपाली</option>
              <option value="si">සිංහල</option>
            </select>
            <button className="theme-btn" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
              {theme === 'light' ? '🌙' : '☀️'}
            </button>
            <button className="culture-btn" onClick={() => setUiTheme(uiTheme === 'nepali-theme' ? 'srilankan-theme' : 'nepali-theme')}>
              {uiTheme === 'nepali-theme' ? '🏔️' : '🌴'}
            </button>
          </div>
        </div>
        <div className="header-ornament bottom"></div>
      </header>

      <aside className="cultural-sidebar">
        <nav className="sidebar-nav">
          <Sidebar activeTab="" setActiveTab={() => {}} />
        </nav>
      </aside>
      
      <main className="cultural-main with-sidebar">
        <div className="content-container">
          {children}
        </div>
      </main>

      <footer className="cultural-footer">
        <div className="footer-pattern"></div>
        <div className="footer-content">
          <p>🕉️ संस्कृति - Multilingual Translation & Cultural Bridge 🕉️</p>
          <p>Privacy-Safe • Cultural Heritage Preserved</p>
        </div>
        <div className="footer-pattern bottom"></div>
      </footer>

    </div>
  )
}