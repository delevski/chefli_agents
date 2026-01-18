"""FastAPI application for multi-agent recipe generation."""

import os
import logging
from dotenv import load_dotenv

# Load environment variables FIRST, before any LangChain imports
load_dotenv()

# Configure LangSmith tracing BEFORE importing LangChain modules
# This ensures tracing is enabled from the start
if os.getenv('LANGCHAIN_API_KEY'):
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ.setdefault('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
    os.environ.setdefault('LANGCHAIN_PROJECT', 'chefli-agents')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.models.schemas import MenuInput, RecipeResponse
from src.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log LangSmith status
if os.getenv('LANGCHAIN_TRACING_V2') == 'true':
    logger.info(f"✅ LangSmith tracing ENABLED for project: {os.getenv('LANGCHAIN_PROJECT', 'chefli-agents')}")
else:
    logger.info("⚠️  LangSmith tracing is DISABLED. Set LANGCHAIN_API_KEY to enable.")

# Initialize FastAPI app
app = FastAPI(
    title="Chefli Agents API",
    description="Multi-agent system for recipe generation",
    version="1.0.0"
)

# Get LLM provider from environment (default to openai)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Initialize orchestrator
try:
    orchestrator = Orchestrator(llm_provider=LLM_PROVIDER)
    logger.info(f"Orchestrator initialized with LLM provider: {LLM_PROVIDER}")
except Exception as e:
    logger.error(f"Failed to initialize orchestrator: {e}")
    orchestrator = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chefli Agents API",
        "version": "1.0.0",
        "endpoints": {
            "generate_recipe": "/generate-recipe (POST)"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "llm_provider": LLM_PROVIDER}


@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(menu_input: MenuInput):
    """
    Generate a recipe from menu input.

    Args:
        menu_input: MenuInput object containing menu description

    Returns:
        RecipeResponse with recipe, image prompt, and nutrition data
    """
    if orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Orchestrator not initialized. Check API keys and LLM provider configuration."
        )

    try:
        logger.info(f"Processing menu input: {menu_input.menu[:50]}...")
        
        # Process menu through orchestrator
        response = await orchestrator.process_menu(menu_input.menu)
        
        logger.info(f"Successfully generated recipe: {response.recipe.dish_name}")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
