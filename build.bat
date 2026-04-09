@echo off
echo ==========================================
echo   Building Buddy AI — .exe
echo ==========================================

:: Install PyInstaller if not already installed
pip install pyinstaller --quiet

:: Build the exe
:: --onefile        = single .exe file
:: --windowed       = no console window (GUI only)
:: --add-data       = bundle intents.json and trained model
:: --name           = output file name

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "BuddyAI" ^
  --add-data "intents.json;." ^
  --add-data "trained_model.pth;." ^
  app.py

echo.
echo ==========================================
if exist "dist\BuddyAI.exe" (
    echo   SUCCESS! Your exe is at:
    echo   dist\BuddyAI.exe
) else (
    echo   Something went wrong. Check the output above.
)
echo ==========================================
pause
