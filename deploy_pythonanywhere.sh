#!/bin/bash
# Complete PythonAnywhere Deployment Script for ordi1985
# Copy and paste these commands in PythonAnywhere Bash console

echo "🚀 Starting PythonAnywhere Deployment..."
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
cd ~
git clone https://github.com/delevski/chefli_agents.git
cd chefli_agents

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
python3.10 -m venv venv

# Step 3: Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip

# Step 5: Install dependencies
echo ""
echo "Step 5: Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python -c "from src.main import app; print('✅ App imports successfully!')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create .env file in Files tab at /home/ordi1985/chefli_agents/.env"
echo "2. Configure web app in Web tab"
echo "3. Copy WSGI configuration from pythonanywhere_wsgi_ordi1985.py"
