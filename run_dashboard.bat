@echo off
echo Killing existing Streamlit processes...
powershell -Command "Get-WmiObject Win32_Process | Where-Object {$_.CommandLine -like '*streamlit*'} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
timeout /t 2 /nobreak >nul

echo Starting Dashboard on Port 8503...
python -m streamlit run dashboard.py
