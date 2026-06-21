#!/bin/bash
set -e

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Streamlit App..."
streamlit run app.py --server.headless true
