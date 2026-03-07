@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Building transmitter.exe...
pyinstaller --onefile --console --name transmitter ^
    --hidden-import=serial ^
    --hidden-import=lz4.block ^
    transmitter.py

echo.
if exist dist\transmitter.exe (
    echo Build successful! Output: dist\transmitter.exe
    copy run.bat dist\run.bat >nul
    echo Copied run.bat to dist\
) else (
    echo Build FAILED!
)
pause
