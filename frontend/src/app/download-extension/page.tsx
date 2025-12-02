'use client'

'use client'

import Layout from '@/components/Layout'

export default function DownloadExtension() {
  const launchExtension = async () => {
    try {
      // Use fetch to call a local endpoint that will launch the extension
      const response = await fetch('/api/launch-extension', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (response.ok) {
        alert('Extension launched successfully! Look for the floating circular button on your screen.')
      } else {
        throw new Error('Failed to launch extension')
      }
    } catch (error) {
      alert('Failed to launch extension. Please try again.')
    }
  }

  return (
    <Layout>
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-4xl font-bold mb-8" style={{color: '#8b4513'}}>Get Extension</h1>
        
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8" style={{border: '2px solid #daa520'}}>
          <h2 className="text-2xl font-semibold mb-4" style={{color: '#8b4513'}}>Floating Translator Extension</h2>
          <p className="text-lg text-gray-600 mb-6">A floating translator that stays on top of all applications for instant translation</p>
          
          <button 
            onClick={launchExtension}
            className="px-12 py-4 rounded-lg font-bold text-white text-xl transition-all duration-300 hover:scale-105 mb-6"
            style={{
              background: 'linear-gradient(135deg, #8b4513, #daa520)',
              boxShadow: '0 6px 20px rgba(139, 69, 19, 0.4)'
            }}
          >
            🚀 Launch Extension
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6" style={{border: '1px solid #daa520'}}>
            <h3 className="text-xl font-semibold mb-3" style={{color: '#8b4513'}}>✨ Features</h3>
            <ul className="text-left text-gray-600 space-y-2">
              <li>• Floating circular button</li>
              <li>• Always on top</li>
              <li>• Drag to move anywhere</li>
              <li>• Instant translation</li>
              <li>• Cultural design theme</li>
            </ul>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6" style={{border: '1px solid #daa520'}}>
            <h3 className="text-xl font-semibold mb-3" style={{color: '#8b4513'}}>🔧 Requirements</h3>
            <ul className="text-left text-gray-600 space-y-2">
              <li>• Python 3.7+</li>
              <li>• tkinter (usually included)</li>
              <li>• requests library</li>
              <li>• pyperclip library</li>
              <li>• Pillow library</li>
            </ul>
          </div>
        </div>
      </div>
    </Layout>
  )
}