#!/bin/bash
# Script to run the Chefli Agents API server

echo "Starting Chefli Agents API..."
echo "================================"

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create .env file with your API keys."
    echo "You can copy .env.example to .env and add your keys."
    exit 1
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API docs will be available at http://localhost:8000/docs"
echo ""
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
