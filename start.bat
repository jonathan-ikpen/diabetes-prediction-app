@echo off
setlocal

:: Architecture Mapping:
:: 1. Checks if the virtual environment exists. If not, creates it.
:: 2. Activates the virtual environment.
:: 3. Installs dependencies from requirements.txt.
:: 4. Runs the Flask web application.

:: Defensive Matrix (Edge Cases Handled):
:: 1. Missing Python installation: Handled by verifying 'python --version' before proceeding.
:: 2. Virtual environment corruption: Checks for the existence of the activation script.
:: 3. Execution context: 'setlocal' ensures environment variables do not leak into the global scope.
:: 4. User path spaces: Quotes are used around file paths to prevent space-in-path execution errors.

echo Checking for Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.9 or newer and try again.
    pause
    exit /b 1
)

echo Checking for virtual environment...
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting the Diabetes Risk Prediction Web App...
echo.
python run.py

pause
