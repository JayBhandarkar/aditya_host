@echo off
echo Starting Translation Website...
echo.

echo Starting Backend Server...
start "Backend" cmd /k "cd backend && python main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Website is starting up...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause