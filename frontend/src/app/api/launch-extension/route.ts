import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'

export async function POST(request: NextRequest) {
  try {
    const extensionPath = 'c:\\Users\\chinm\\Documents\\sih round 2\\extension'
    const command = `cd "${extensionPath}" && python floating_translator.py`
    
    // Execute the command in the background
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error('Extension launch error:', error)
      } else {
        console.log('Extension launched successfully')
      }
    })
    
    return NextResponse.json({ success: true, message: 'Extension launched' })
  } catch (error) {
    console.error('Failed to launch extension:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to launch extension' },
      { status: 500 }
    )
  }
}