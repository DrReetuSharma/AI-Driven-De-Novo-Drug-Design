#!/bin/bash

chmod +x run_api.sh

# Activate the Python virtual environment (if using one)
source venv/bin/activate

# Run the API server
python src/api.py
