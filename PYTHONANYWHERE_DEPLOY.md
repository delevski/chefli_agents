# PythonAnywhere Deployment Guide

## Prerequisites

1. Sign up for a free PythonAnywhere account: https://www.pythonanywhere.com
2. Free tier includes:
   - 1 web app
   - 512MB disk space
   - Limited CPU (enough for testing)

## Step 1: Upload Your Code

### Option A: Using Git (Recommended)

1. In PythonAnywhere dashboard, open a **Bash console**
2. Clone your repository:
```bash
cd ~
git clone https://github.com/delevski/chefli_agents.git
cd chefli_agents
```

### Option B: Using Files Tab

1. Go to **Files** tab in PythonAnywhere dashboard
2. Create directory: `chefli-agents`
3. Upload all files manually (not recommended, use Git)

## Step 2: Set Up Virtual Environment

In the Bash console:
```bash
cd ~/chefli_agents
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Configure Environment Variables

1. Go to **Files** tab
2. Navigate to `chefli_agents` directory
3. Create `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=chefli-agents
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LLM_PROVIDER=openai
```

## Step 4: Update WSGI Configuration

1. Go to **Web** tab in PythonAnywhere dashboard
2. Click **Add a new web app**
3. Choose **Manual configuration**
4. Select **Python 3.10** (or latest available)
5. Click **Next** → **Finish**

6. Click on the web app name to edit it
7. In the **WSGI configuration file** section, click on the file path
8. Replace the content with:

```python
# This file is used by PythonAnywhere to serve your FastAPI app

import sys
import os

# Add your project directory to the path
# Replace 'yourusername' with your PythonAnywhere username
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

**Important:** Replace `yourusername` with your actual PythonAnywhere username!

## Step 5: Configure Web App Settings

In the **Web** tab, configure:

1. **Source code**: `/home/yourusername/chefli_agents`
2. **Working directory**: `/home/yourusername/chefli_agents`
3. **WSGI configuration file**: Use the file you edited above

## Step 6: Set Up Static Files (if needed)

FastAPI doesn't need static files, but you can skip this section.

## Step 7: Reload Web App

1. Go to **Web** tab
2. Click the green **Reload** button
3. Your app should now be live!

## Step 8: Access Your App

Your app will be available at:
```
https://yourusername.pythonanywhere.com
```

## Troubleshooting

### If app doesn't start:

1. **Check logs**:
   - Go to **Web** tab → **Error log**
   - Look for errors and fix them

2. **Check virtual environment**:
   ```bash
   cd ~/chefli_agents
   source venv/bin/activate
   python -c "from src.main import app; print('OK')"
   ```

3. **Check environment variables**:
   ```bash
   cd ~/chefli_agents
   source venv/bin/activate
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
   ```

4. **Common issues**:
   - Wrong username in WSGI file
   - Virtual environment not activated
   - Missing dependencies
   - Environment variables not set

### Update Your App

1. Pull latest changes:
   ```bash
   cd ~/chefli_agents
   git pull
   ```

2. Update dependencies if needed:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Reload web app in **Web** tab

## Free Tier Limitations

- **CPU**: Limited (may be slow for LLM calls)
- **Disk**: 512MB
- **Web requests**: Limited per day
- **Always-on**: Not available (app sleeps after inactivity)

## Important Notes

- PythonAnywhere free tier has CPU limits - LLM calls may be slow
- App sleeps when inactive - first request after sleep may be slow
- Consider upgrading to paid plan ($5/month) for better performance

## Testing Your Deployment

```bash
curl https://yourusername.pythonanywhere.com/health
```

Or visit in browser:
```
https://yourusername.pythonanywhere.com/docs
```
