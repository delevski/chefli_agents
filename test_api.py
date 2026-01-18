"""Simple test script for the Chefli Agents API."""

import requests
import json
import sys

def test_api(menu_item="Chocolate Chip Cookies"):
    """Test the recipe generation API."""
    url = "http://localhost:8000/generate-recipe"
    payload = {"menu": menu_item}
    
    print(f"Testing API with menu item: {menu_item}")
    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\n" + "="*50 + "\n")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Recipe generated:\n")
            print(f"Dish Name: {result['recipe']['dish_name']}")
            print(f"\nIngredients ({len(result['recipe']['ingredients'])}):")
            for ing in result['recipe']['ingredients']:
                print(f"  - {ing['name']}: {ing['quantity']}")
            print(f"\nInstructions ({len(result['recipe']['instructions'])} steps):")
            for i, step in enumerate(result['recipe']['instructions'], 1):
                print(f"  {i}. {step}")
            print(f"\nNutrition:")
            print(f"  Calories: {result['nutrition']['calories']:.1f}")
            print(f"  Protein: {result['nutrition']['protein']:.1f}g")
            print(f"  Carbohydrates: {result['nutrition']['carbohydrates']:.1f}g")
            print(f"\nImage Prompt (first 200 chars):")
            print(f"  {result['image_prompt'][:200]}...")
            return True
        else:
            print(f"❌ ERROR: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API.")
        print("Make sure the server is running on http://localhost:8000")
        print("\nStart the server with:")
        print("  python -m src.main")
        print("or")
        print("  uvicorn src.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("Chefli Agents API Test")
    print("="*50 + "\n")
    
    # Test health endpoint first
    print("1. Testing health endpoint...")
    if not test_health():
        sys.exit(1)
    
    print("\n" + "="*50 + "\n")
    
    # Test recipe generation
    print("2. Testing recipe generation...")
    menu_item = sys.argv[1] if len(sys.argv) > 1 else "Chocolate Chip Cookies"
    success = test_api(menu_item)
    
    if success:
        print("\n" + "="*50)
        print("✅ All tests passed!")
        print("="*50)
    else:
        sys.exit(1)
