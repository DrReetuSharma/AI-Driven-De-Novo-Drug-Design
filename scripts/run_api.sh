#!/bin/bash

chmod +x run_api.sh

# Activate the Python virtual environment (if using one)
source venv/bin/activate

# Run the API server
python src/api.py

./run_api.sh

curl -X POST "http://0.0.0.0:8000/generate" -H "Content-Type: application/json" -d '{"num_molecules": 10}'

