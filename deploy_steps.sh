#!/bin/bash
# PythonAnywhere Deployment Steps
# Copy and paste these commands one by one in PythonAnywhere Bash console

echo "Step 1: Clone repository"
git clone https://github.com/delevski/chefli_agents.git
cd chefli_agents

echo "Step 2: Create virtual environment"
python3.10 -m venv venv

echo "Step 3: Activate virtual environment"
source venv/bin/activate

echo "Step 4: Upgrade pip"
pip install --upgrade pip

echo "Step 5: Install dependencies"
pip install -r requirements.txt

echo "Step 6: Verify installation"
python -c "from src.main import app; print('✅ App imports successfully!')"

echo ""
echo "✅ Setup complete!"
echo "Next: Create .env file and configure web app"
