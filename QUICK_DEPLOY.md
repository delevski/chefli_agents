# Quick PythonAnywhere Deployment Guide

## Step 1: Clone Repository

In PythonAnywhere **Bash console**, run:
```bash
git clone https://github.com/delevski/chefli_agents.git
cd chefli_agents
```

## Step 2: Set Up Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
python -c "from src.main import app; print('✅ App imports successfully!')"
```

## Step 4: Create .env File

1. Go to **Files** tab in PythonAnywhere
2. Navigate to `/home/yourusername/chefli_agents/`
3. Click **New file** → Name it `.env`
4. Add this content (replace with your actual keys):

```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=chefli-agents
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LLM_PROVIDER=openai
```

**Important:** Replace `yourusername` with your actual PythonAnywhere username!

## Step 5: Configure Web App

1. Go to **Web** tab
2. Click **Add a new web app**
3. Choose **Manual configuration**
4. Select **Python 3.10** (or latest available)
5. Click **Next** → **Finish**

## Step 6: Configure WSGI File

1. In **Web** tab, click on your web app name
2. Find **WSGI configuration file** section
3. Click on the file path (usually `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
4. Replace ALL content with:

```python
# This file is used by PythonAnywhere to serve your FastAPI app
# IMPORTANT: Replace 'yourusername' with your actual PythonAnywhere username!

import sys
import os

# Add your project directory to the path
path = '/home/yourusername/chefli_agents'
if path not in sys.path:
    sys.path.insert(0, path)

# Change to your project directory
os.chdir(path)

# Activate virtual environment
activate_this = '/home/yourusername/chefli_agents/venv/bin/activate_this.py'
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

# Import your FastAPI app
from src.main import app

# PythonAnywhere expects a variable called 'application'
application = app
```

**CRITICAL:** Replace both instances of `yourusername` with your actual PythonAnywhere username!

## Step 7: Set Source Code and Working Directory

Still in **Web** tab, configure:

1. **Source code**: `/home/yourusername/chefli_agents`
2. **Working directory**: `/home/yourusername/chefli_agents`

(Replace `yourusername` with your actual username)

## Step 8: Reload Web App

1. Scroll to top of **Web** tab
2. Click the green **Reload** button
3. Wait for reload to complete

## Step 9: Test Your App

Your app will be available at:
```
https://yourusername.pythonanywhere.com
```

Test endpoints:
- Health: `https://yourusername.pythonanywhere.com/health`
- Docs: `https://yourusername.pythonanywhere.com/docs`
- API: `https://yourusername.pythonanywhere.com/generate-recipe`

## Troubleshooting

### If reload fails:

1. Check **Error log** in Web tab
2. Common issues:
   - Wrong username in WSGI file
   - Virtual environment not activated
   - Missing dependencies
   - Environment variables not set

### Test in Bash console:

```bash
cd ~/chefli_agents
source venv/bin/activate
python -c "from src.main import app; print('OK')"
```

### Check environment variables:

```bash
cd ~/chefli_agents
source venv/bin/activate
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## Need Help?

Check the full guide: `PYTHONANYWHERE_DEPLOY.md`
