"""Chef Agent - Generates recipes from menu input."""

import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.models.schemas import Recipe, Ingredient


class ChefAgent:
    """Agent that generates recipes from menu descriptions."""

    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize Chef Agent.

        Args:
            llm_provider: LLM provider to use ('openai' or 'anthropic')
        """
        self.llm_provider = llm_provider.lower()
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        if self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                api_key=api_key
            )
        else:  # Default to OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                api_key=api_key
            )

    async def generate_recipe(self, menu: str, language: str = "English") -> Recipe:
        """
        Generate a recipe from menu input.

        Args:
            menu: Menu item or description
            language: Language to respond in (e.g., "Hebrew", "English", etc.)

        Returns:
            Recipe object with dish name, ingredients, and instructions
        """
        language_instruction = f"IMPORTANT: Respond entirely in {language}. All text including dish name, ingredients, and instructions must be in {language}."
        
        # Build system prompt with properly escaped JSON example
        json_example = '{{\n    "dish_name": "Name of the dish",\n    "ingredients": [\n        {{"name": "ingredient name", "quantity": "amount and unit"}},\n        ...\n    ],\n    "instructions": [\n        "Step 1 instruction",\n        "Step 2 instruction",\n        ...\n    ]\n}}'
        
        system_prompt = f"""You are an expert chef. Generate a detailed recipe based on the menu item provided.
{language_instruction}

CRITICAL RULES:
- Do NOT invent recipes, steps, or data. Use only real, accurate information.
- If information is missing or uncertain, state it explicitly (e.g., "approximately", "if available").
- Writing must be concise, clear, and practical. No storytelling, slogans, or marketing language.
- Focus strictly on accuracy, usability, and real-world execution.
- Use all or most of the provided ingredients when possible, but only if they are necessary for the dish. Do not force ingredients that don't belong.

The recipe should include:
1. A clear dish name
2. A complete list of ingredients with quantities (use all/most provided ingredients when appropriate)
3. Step-by-step cooking instructions (concise and practical)

Format your response as JSON with the following structure:
{json_example}

Remember: All text must be in {language}. Be accurate, practical, and avoid any marketing language."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Menu item: {menu}")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({"menu": menu})

        # Parse the response
        import json
        try:
            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            recipe_data = json.loads(content)
            
            # Convert to Recipe model
            ingredients = [
                Ingredient(name=ing["name"], quantity=ing["quantity"])
                for ing in recipe_data["ingredients"]
            ]
            
            return Recipe(
                dish_name=recipe_data["dish_name"],
                ingredients=ingredients,
                instructions=recipe_data["instructions"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse recipe response: {e}")
