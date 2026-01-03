from typing import List, Dict, Any, Optional
import json
import os

def load_food_data(file_path: str) -> List[Dict]:
    """Load food data from JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)
        print(f"📝 Project directory: {os.getcwd()}")
        print(f"📝 Current directory: {script_dir}")
        print(f"📝 Full file path: {file_path}")
        print("======")
        print(f"⏳ Loading food data from {file_path}")

        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                food_data = json.load(file)
            print("✅ Data added to a variable.")
        else:
            print(f"❌ File not found at: {file_path}")

        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])
            
            # Ensure required fields exist
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0
            
            # Extract taste features from nested food_features if available
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                item['taste_profile'] = ', '.join(taste_features)
            else:
                item['taste_profile'] = ''
        
        print(f"✅ Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data
        
    except Exception as e:
        print(f"Error loading food data: {e}")
        return []
