"""Pydantic models for request and response schemas."""

from typing import List
from pydantic import BaseModel, Field


class MenuInput(BaseModel):
    """Input schema for menu."""
    menu: str = Field(..., description="Menu item or description")


class Ingredient(BaseModel):
    """Ingredient model."""
    name: str = Field(..., description="Ingredient name")
    quantity: str = Field(..., description="Quantity and unit")


class Recipe(BaseModel):
    """Recipe structure."""
    dish_name: str = Field(..., description="Name of the dish")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients")
    instructions: List[str] = Field(..., description="Step-by-step cooking instructions")


class Nutrition(BaseModel):
    """Nutrition values."""
    calories: float = Field(..., description="Total calories")
    protein: float = Field(..., description="Protein in grams")
    carbohydrates: float = Field(..., description="Carbohydrates in grams")
    fiber: float = Field(..., description="Fiber in grams")
    fats: float = Field(..., description="Fats in grams")


class RecipeResponse(BaseModel):
    """Final JSON response structure."""
    recipe: Recipe = Field(..., description="Generated recipe")
    image_prompt: str = Field(..., description="Image generation prompt")
    image_url: str = Field(None, description="URL of the generated image")
    nutrition: Nutrition = Field(..., description="Nutritional information")
