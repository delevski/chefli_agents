"""Orchestrator - Coordinates multi-agent workflow."""

import asyncio
import re
import hashlib
from functools import lru_cache
from src.agents.chef_agent import ChefAgent
from src.agents.design_agent import DesignAgent
from src.agents.nutrition_agent import NutritionAgent
from src.models.schemas import RecipeResponse, Recipe


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        Language name (e.g., 'Hebrew', 'English', 'Spanish', etc.)
    """
    # Simple heuristic-based detection
    # Check for Hebrew characters (Unicode range: \u0590-\u05FF)
    if re.search(r'[\u0590-\u05FF]', text):
        return "Hebrew"
    # Check for Arabic characters
    elif re.search(r'[\u0600-\u06FF]', text):
        return "Arabic"
    # Check for Chinese/Japanese/Korean characters
    elif re.search(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', text):
        return "Asian"
    # Check for Cyrillic (Russian, etc.)
    elif re.search(r'[\u0400-\u04FF]', text):
        return "Cyrillic"
    # Check for common European languages with special characters
    elif re.search(r'[àáâãäåæçèéêëìíîïñòóôõöøùúûüý]', text, re.IGNORECASE):
        return "European"
    # Default to English
    else:
        return "English"


class Orchestrator:
    """Orchestrates the multi-agent recipe generation workflow."""

    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize Orchestrator.

        Args:
            llm_provider: LLM provider to use ('openai' or 'anthropic')
        """
        self.llm_provider = llm_provider
        self.chef_agent = ChefAgent(llm_provider)
        self.design_agent = DesignAgent(llm_provider)
        self.nutrition_agent = NutritionAgent(llm_provider)

    async def process_menu(self, menu: str) -> RecipeResponse:
        """
        Process menu input through the multi-agent system.

        Args:
            menu: Menu item or description

        Returns:
            RecipeResponse with recipe, image prompt, and nutrition data
        """
        # Detect the language of the menu input
        detected_language = detect_language(menu)
        
        # Step 1: Generate recipe using Chef Agent (with language context)
        recipe = await self.chef_agent.generate_recipe(menu, language=detected_language)

        # Step 2: Process recipe in parallel with Design and Nutrition agents (with language context)
        image_task = self.design_agent.generate_image(recipe, language=detected_language)
        nutrition_task = self.nutrition_agent.calculate_nutrition(recipe, language=detected_language)

        # Run both agents in parallel
        image_result, nutrition = await asyncio.gather(
            image_task,
            nutrition_task
        )

        # Step 3: Combine results
        return RecipeResponse(
            recipe=recipe,
            image_prompt=image_result.get("image_prompt", ""),
            image_url=image_result.get("image_url"),
            nutrition=nutrition
        )
