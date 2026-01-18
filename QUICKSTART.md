# Quick Start Guide

## 🚀 Running the Server

### Step 1: Add Your API Key

Edit the `.env` file and add your OpenAI or Anthropic API key:

```bash
# For OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-api-key-here

# OR for Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here
```

### Step 2: Start the Server

**Option A: Using the startup script**
```bash
./run_server.sh
```

**Option B: Using uvicorn directly**
```bash
source venv/bin/activate
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Option C: Using Python**
```bash
source venv/bin/activate
python -m src.main
```

### Step 3: Test the API

Once the server is running, you have several options:

#### A. Interactive API Docs (Easiest)
Open your browser and go to: **http://localhost:8000/docs**

Click "Try it out" on the `/generate-recipe` endpoint, enter:
```json
{
  "menu": "Chocolate Chip Cookies"
}
```
Click "Execute" to see the result!

#### B. Using the Test Script
```bash
source venv/bin/activate
python test_api.py "Grilled Salmon"
```

#### C. Using cURL
```bash
curl -X POST "http://localhost:8000/generate-recipe" \
  -H "Content-Type: application/json" \
  -d '{"menu": "Pasta Carbonara"}'
```

#### D. Health Check
```bash
curl http://localhost:8000/health
```

## 📝 Example Response

```json
{
  "recipe": {
    "dish_name": "Chocolate Chip Cookies",
    "ingredients": [
      {"name": "all-purpose flour", "quantity": "2 1/4 cups"},
      {"name": "butter", "quantity": "1 cup"},
      ...
    ],
    "instructions": [
      "Preheat oven to 375°F",
      "Mix dry ingredients",
      ...
    ]
  },
  "image_prompt": "A beautiful plate of warm chocolate chip cookies...",
  "nutrition": {
    "calories": 150.0,
    "protein": 2.0,
    "carbohydrates": 20.0
  }
}
```

## ⚠️ Troubleshooting

- **"API key not set"**: Make sure you've added your API key to `.env` file
- **"Connection refused"**: Make sure the server is running
- **Import errors**: Make sure virtual environment is activated: `source venv/bin/activate`
