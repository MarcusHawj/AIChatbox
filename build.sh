#!/bin/bash
echo "=========================================="
echo "  Building Buddy AI — executable"
echo "=========================================="

# Install PyInstaller if needed
pip install pyinstaller --quiet

# Detect OS for correct path separator
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    SEP=";"
else
    SEP=":"
fi

pyinstaller \
  --onefile \
  --windowed \
  --name "BuddyAI" \
  --add-data "intents.json${SEP}." \
  --add-data "trained_model.pth${SEP}." \
  app.py

echo ""
echo "=========================================="
if [ -f "dist/BuddyAI" ]; then
    echo "  SUCCESS! Your app is at: dist/BuddyAI"
elif [ -f "dist/BuddyAI.exe" ]; then
    echo "  SUCCESS! Your exe is at: dist/BuddyAI.exe"
else
    echo "  Something went wrong. Check output above."
fi
echo "=========================================="
