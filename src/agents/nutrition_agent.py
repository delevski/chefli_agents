"""Nutrition Agent - Calculates nutritional values from recipes."""

import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.models.schemas import Recipe, Nutrition


class NutritionAgent:
    """Agent that calculates nutritional values from recipes."""

    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize Nutrition Agent.

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
                model="claude-3-haiku-20240307",  # Faster Anthropic model
                temperature=0.1,  # Lower temperature for faster responses
                api_key=api_key
            )
        else:  # Default to OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model="gpt-4o-mini",  # Faster model for calculations
                temperature=0.1,  # Lower temperature for faster, more deterministic responses
                api_key=api_key
            )

    def _recipe_hash(self, recipe: Recipe) -> str:
        """Create a hash of recipe for caching."""
        recipe_str = f"{recipe.dish_name}:{','.join([f'{ing.name}:{ing.quantity}' for ing in recipe.ingredients])}"
        return hashlib.md5(recipe_str.encode()).hexdigest()

    async def calculate_nutrition(self, recipe: Recipe, language: str = "English") -> Nutrition:
        """
        Calculate nutritional values from recipe ingredients.

        Args:
            recipe: Recipe object
            language: Language to respond in (e.g., "Hebrew", "English", etc.)

        Returns:
            Nutrition object with calories, protein, and carbohydrates
        """
        language_instruction = f"IMPORTANT: Respond entirely in {language}. All text must be in {language}."
        
        # Build system prompt with properly escaped JSON example
        json_example = '{{\n    "calories": <total calories as float>,\n    "protein": <total protein in grams as float>,\n    "carbohydrates": <total carbohydrates in grams as float>,\n    "fiber": <total fiber in grams as float>,\n    "fats": <total fats in grams as float>\n}}'
        
        system_prompt = f"""Calculate nutritional values for the recipe.
{language_instruction}

Rules:
- Use real values from nutritional databases only. No estimates.
- If uncertain, use 0 and note it.
- Calculate totals for entire recipe.

Return JSON:
{json_example}

Be precise. {language} only. No marketing language."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Dish: {dish_name}
            
Ingredients:
{ingredients}

Calculate the nutritional values for this recipe:""")
        ])

        # Format ingredients
        ingredients_text = "\n".join([
            f"- {ing.name}: {ing.quantity}" for ing in recipe.ingredients
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "dish_name": recipe.dish_name,
            "ingredients": ingredients_text
        })

        # Parse the response
        try:
            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            nutrition_data = json.loads(content)
            
            return Nutrition(
                calories=float(nutrition_data["calories"]),
                protein=float(nutrition_data["protein"]),
                carbohydrates=float(nutrition_data["carbohydrates"]),
                fiber=float(nutrition_data.get("fiber", 0.0)),
                fats=float(nutrition_data.get("fats", 0.0))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse nutrition response: {e}")
