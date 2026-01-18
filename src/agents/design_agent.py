"""Design Agent - Generates images from recipes."""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import OpenAI

from src.models.schemas import Recipe


class DesignAgent:
    """Agent that generates image prompts from recipes."""

    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize Design Agent.

        Args:
            llm_provider: LLM provider to use ('openai' or 'anthropic')
        """
        self.llm_provider = llm_provider.lower()
        self.llm = self._initialize_llm()
        # Initialize OpenAI client for DALL-E image generation
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        if self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return ChatAnthropic(
                model="claude-3-haiku-20240307",  # Faster Anthropic model
                temperature=0.5,  # Lower temperature for faster responses
                api_key=api_key
            )
        else:  # Default to OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model="gpt-4o-mini",  # Faster model for image prompt generation
                temperature=0.5,  # Lower temperature for faster responses
                api_key=api_key
            )

    async def generate_image_prompt(self, recipe: Recipe, language: str = "English") -> str:
        """
        Generate an image prompt from a recipe.

        Args:
            recipe: Recipe object
            language: Language to respond in (e.g., "Hebrew", "English", etc.)

        Returns:
            Detailed image generation prompt
        """
        language_instruction = f"IMPORTANT: Respond entirely in {language}. The image prompt must be written in {language}."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Create an image generation prompt for the dish.
{language_instruction}

Rules:
- Concise, practical. No storytelling or marketing.
- Describe only what's in the recipe.
- Include: appearance, colors, textures, plating, lighting, garnishes (if in recipe).

Return only the prompt text in {language}. No formatting."""),
            ("human", """Dish: {dish_name}
            
Ingredients: {ingredients}
            
Instructions: {instructions}
            
Generate a detailed image prompt for this dish:""")
        ])

        # Format ingredients and instructions
        ingredients_text = "\n".join([
            f"- {ing.name}: {ing.quantity}" for ing in recipe.ingredients
        ])
        instructions_text = "\n".join([
            f"{i+1}. {step}" for i, step in enumerate(recipe.instructions)
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "dish_name": recipe.dish_name,
            "ingredients": ingredients_text,
            "instructions": instructions_text
        })

        return response.content.strip()

    async def generate_image(self, recipe: Recipe, language: str = "English") -> dict:
        """
        Generate an actual image from a recipe using DALL-E.

        Args:
            recipe: Recipe object
            language: Language to respond in (e.g., "Hebrew", "English", etc.)

        Returns:
            Dictionary with image_url and image_prompt
        """
        # First generate the prompt
        image_prompt = await self.generate_image_prompt(recipe, language=language)
        
        # Generate image using DALL-E if OpenAI client is available
        if self.openai_client is None:
            return {
                "image_url": None,
                "image_prompt": image_prompt,
                "error": "OpenAI API key not configured"
            }
        
        try:
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            return {
                "image_url": image_url,
                "image_prompt": image_prompt
            }
        except Exception as e:
            # Fallback to just returning the prompt if image generation fails
            return {
                "image_url": None,
                "image_prompt": image_prompt,
                "error": str(e)
            }
