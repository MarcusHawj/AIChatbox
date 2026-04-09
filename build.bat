@echo off
echo ==========================================
echo   Building Buddy AI — .exe
echo ==========================================

pip install pyinstaller --quiet

pyinstaller ^
  --onefile ^
  --windowed ^
  --name "BuddyAI" ^
  --add-data "intents.json;." ^
  --add-data "trained_model.pth;." ^
  --hidden-import customtkinter ^
  --exclude-module tensorflow ^
  --exclude-module keras ^
  --exclude-module tensorboard ^
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