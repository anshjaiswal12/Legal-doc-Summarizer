#!/bin/bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading NLTK punkt..."
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "Downloading spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "Environment setup complete."
