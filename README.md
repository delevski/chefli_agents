# Chefli Agents - Multi-Agent Recipe Generation System

A production-ready multi-agent system built with LangChain and FastAPI that generates complete recipes from menu inputs, including image prompts and nutritional information.

## Features

- **Chef Agent**: Generates detailed recipes with ingredients and step-by-step instructions
- **Design Agent**: Creates image generation prompts for visual representation
- **Nutrition Agent**: Calculates nutritional values (calories, protein, carbohydrates)
- **Parallel Processing**: Design and Nutrition agents run simultaneously for efficiency
- **RESTful API**: FastAPI endpoint for easy integration

## Architecture

```
Menu Input → Chef Agent → Recipe
                          ↓
                    ┌─────┴─────┐
                    │  Parallel │
                    └─────┬─────┘
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
    Design Agent              Nutrition Agent
            ↓                           ↓
    Image Prompt              Nutrition Values
            └─────────────┬─────────────┘
                          ↓
                    JSON Response
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

Or for Anthropic:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Running Locally

1. **Start the FastAPI server**:
```bash
python -m src.main
```

Or using uvicorn directly:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

2. **The API will be available at**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## API Usage

### Generate Recipe

**Endpoint**: `POST /generate-recipe`

**Request Body**:
```json
{
  "menu": "Grilled Salmon with Lemon Butter Sauce"
}
```

**Response**:
```json
{
  "recipe": {
    "dish_name": "Grilled Salmon with Lemon Butter Sauce",
    "ingredients": [
      {
        "name": "salmon fillets",
        "quantity": "4 (6 oz each)"
      },
      {
        "name": "butter",
        "quantity": "4 tablespoons"
      },
      ...
    ],
    "instructions": [
      "Preheat grill to medium-high heat",
      "Season salmon fillets with salt and pepper",
      ...
    ]
  },
  "image_prompt": "A beautifully plated grilled salmon fillet with golden-brown grill marks, drizzled with creamy lemon butter sauce...",
  "nutrition": {
    "calories": 450.0,
    "protein": 35.0,
    "carbohydrates": 2.0
  }
}
```

### Example cURL Command

```bash
curl -X POST "http://localhost:8000/generate-recipe" \
  -H "Content-Type: application/json" \
  -d '{"menu": "Chocolate Chip Cookies"}'
```

### Example Python Request

```python
import requests

url = "http://localhost:8000/generate-recipe"
payload = {"menu": "Chocolate Chip Cookies"}
response = requests.post(url, json=payload)
print(response.json())
```

## Project Structure

```
chefli-agents/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── chef_agent.py      # Recipe generation agent
│   │   ├── design_agent.py    # Image prompt generation agent
│   │   └── nutrition_agent.py # Nutrition calculation agent
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   ├── orchestrator.py        # Coordinates agent workflow
│   └── main.py                # FastAPI application
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

### LLM Provider

The system supports two LLM providers:

1. **OpenAI** (default): Uses GPT-4 Turbo
   - Set `LLM_PROVIDER=openai`
   - Requires `OPENAI_API_KEY`

2. **Anthropic**: Uses Claude 3 Sonnet
   - Set `LLM_PROVIDER=anthropic`
   - Requires `ANTHROPIC_API_KEY`

## Development

### Code Style

The project follows Python best practices:
- Type hints throughout
- Pydantic models for validation
- Async/await for concurrent operations
- Modular agent architecture

### Testing

Test the API using the interactive docs at http://localhost:8000/docs or use the example curl commands above.

## Production Deployment

For production deployment:

1. Set proper environment variables
2. Use a production ASGI server like Gunicorn with Uvicorn workers:
```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. Configure reverse proxy (nginx, etc.)
4. Set up monitoring and logging
5. Use environment-specific configuration files

## License

MIT License
