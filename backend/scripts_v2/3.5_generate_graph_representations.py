"""
Generate graph representations for recipes using regex and rule-based parsing.
No AI API calls - fast batch processing for large datasets.

Targets the 'recipes_backup' table.
"""

import os
import json
import re
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target table
TABLE_NAME = "recipes_duplicate"

# Cooking action verbs - categorized by type
COOKING_ACTIONS = {
    # Heat-based
    'heat': ['heat', 'warm', 'preheat'],
    'boil': ['boil', 'simmer', 'blanch', 'parboil'],
    'fry': ['fry', 'sauté', 'saute', 'pan-fry', 'deep-fry', 'stir-fry', 'sear'],
    'bake': ['bake', 'roast', 'broil', 'toast', 'grill', 'barbecue', 'bbq'],
    'steam': ['steam'],
    'microwave': ['microwave', 'nuke'],
    
    # Mixing
    'mix': ['mix', 'combine', 'blend', 'stir', 'whisk', 'beat', 'fold', 'incorporate'],
    'cream': ['cream', 'whip'],
    
    # Cutting
    'chop': ['chop', 'dice', 'mince', 'cube', 'julienne'],
    'slice': ['slice', 'cut'],
    'shred': ['shred', 'grate', 'zest'],
    
    # Preparation
    'add': ['add', 'pour', 'sprinkle', 'drizzle', 'top'],
    'season': ['season', 'salt', 'pepper', 'spice'],
    'marinate': ['marinate', 'brine', 'soak'],
    'coat': ['coat', 'bread', 'dredge', 'dust', 'flour'],
    'melt': ['melt', 'dissolve'],
    'drain': ['drain', 'strain', 'rinse'],
    'cool': ['cool', 'chill', 'refrigerate', 'freeze'],
    'knead': ['knead', 'roll', 'flatten', 'press', 'shape', 'form'],
    'spread': ['spread', 'layer', 'arrange'],
    'garnish': ['garnish', 'decorate', 'serve'],
}

# Build reverse lookup: word -> action category
ACTION_LOOKUP = {}
for category, words in COOKING_ACTIONS.items():
    for word in words:
        ACTION_LOOKUP[word] = category


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def extract_action_from_step(step: str) -> str:
    """Extract the primary cooking action from a direction step."""
    step_lower = step.lower()
    
    # Look for action verbs at the beginning of the sentence (most common)
    words = re.findall(r'\b[a-z]+\b', step_lower)
    
    for word in words[:5]:  # Check first 5 words
        if word in ACTION_LOOKUP:
            return ACTION_LOOKUP[word]
    
    # Check entire step for any action
    for word in words:
        if word in ACTION_LOOKUP:
            return ACTION_LOOKUP[word]
    
    return "process"  # Default action


def find_ingredients_in_step(step: str, ner: list) -> list[int]:
    """Find which ingredients (by index) are mentioned in a step."""
    step_lower = step.lower()
    found_indices = []
    
    for idx, ingredient in enumerate(ner):
        # Create pattern for ingredient (handle multi-word ingredients)
        ing_lower = ingredient.lower()
        
        # Check for exact match or partial match
        if ing_lower in step_lower:
            found_indices.append(idx)
        else:
            # Check individual words for longer ingredients
            ing_words = ing_lower.split()
            if len(ing_words) > 1:
                # Check if main word is present
                for word in ing_words:
                    if len(word) > 3 and word in step_lower:
                        found_indices.append(idx)
                        break
    
    return found_indices


def generate_graph_from_recipe(
    recipe_name: str,
    ingredients: list,
    directions: list,
    ner: list
) -> dict:
    """Generate a graph representation using regex and rules."""
    
    nodes = []
    edges = []
    
    # Add ingredient nodes
    ingredient_count = len(ner)
    for idx, ing in enumerate(ner):
        nodes.append({
            "id": idx,
            "type": "ingredient",
            "label": ing,
            "state": "raw"
        })
    
    if not directions:
        # No directions - simple graph connecting all to final
        final_id = ingredient_count
        nodes.append({
            "id": final_id,
            "type": "final",
            "label": recipe_name or "Final Dish",
            "state": "complete"
        })
        for idx in range(ingredient_count):
            edges.append({
                "source": idx,
                "target": final_id,
                "action": "combine",
                "step": 1
            })
        return create_graph_output(nodes, edges, ner, is_fallback=True)
    
    # Track which ingredients have been used
    used_ingredients = set()
    current_node_id = ingredient_count
    previous_intermediate_id = None
    
    # Process each direction step
    for step_idx, step in enumerate(directions):
        if not step or not step.strip():
            continue
            
        # Find ingredients mentioned in this step
        step_ingredients = find_ingredients_in_step(step, ner)
        
        # Extract the cooking action
        action = extract_action_from_step(step)
        
        # Create intermediate node for this step
        is_final = (step_idx == len(directions) - 1)
        
        # Determine node label
        if is_final:
            node_label = recipe_name or "Final Dish"
            node_type = "final"
            node_state = "complete"
        else:
            node_label = f"{action.capitalize()} mixture"
            node_type = "intermediate"
            node_state = action + "ed" if not action.endswith('e') else action + "d"
        
        intermediate_node = {
            "id": current_node_id,
            "type": node_type,
            "label": node_label,
            "state": node_state
        }
        nodes.append(intermediate_node)
        
        # Connect ingredients mentioned in this step
        if step_ingredients:
            for ing_idx in step_ingredients:
                if ing_idx not in used_ingredients:
                    edges.append({
                        "source": ing_idx,
                        "target": current_node_id,
                        "action": action,
                        "step": step_idx + 1
                    })
                    used_ingredients.add(ing_idx)
        
        # Connect from previous intermediate node (if exists)
        if previous_intermediate_id is not None:
            edges.append({
                "source": previous_intermediate_id,
                "target": current_node_id,
                "action": action,
                "step": step_idx + 1
            })
        
        previous_intermediate_id = current_node_id
        current_node_id += 1
    
    # Connect any unused ingredients to the first intermediate node
    first_intermediate_id = ingredient_count
    for idx in range(ingredient_count):
        if idx not in used_ingredients:
            edges.append({
                "source": idx,
                "target": first_intermediate_id,
                "action": "add",
                "step": 1
            })
    
    # Ensure we have a final node
    if not any(n["type"] == "final" for n in nodes):
        final_id = current_node_id
        nodes.append({
            "id": final_id,
            "type": "final",
            "label": recipe_name or "Final Dish",
            "state": "complete"
        })
        if previous_intermediate_id is not None:
            edges.append({
                "source": previous_intermediate_id,
                "target": final_id,
                "action": "serve",
                "step": len(directions)
            })
    
    return create_graph_output(nodes, edges, ner, is_fallback=False)


def create_graph_output(nodes: list, edges: list, ner: list, is_fallback: bool = False) -> dict:
    """Create the final graph output with metadata."""
    ingredient_count = len([n for n in nodes if n.get("type") == "ingredient"])
    intermediate_count = len([n for n in nodes if n.get("type") == "intermediate"])
    final_count = len([n for n in nodes if n.get("type") == "final"])
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "ingredient_count": ingredient_count,
            "intermediate_count": intermediate_count,
            "final_count": final_count,
            "original_ner_count": len(ner),
            "step_count": max([e.get("step", 0) for e in edges] + [0]),
            "generation_method": "regex",
            "fallback": is_fallback
        }
    }


def parse_json_field(value) -> list:
    """Parse a JSON field that might be a string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return []
    return []


def fetch_recipes(supabase: Client, batch_size: int = 1000, limit: int = None) -> list[dict]:
    """Fetch all recipes from the backup table."""
    print(f"📂 Fetching recipes from {TABLE_NAME}...")
    
    all_recipes = []
    offset = 0
    
    while True:
        response = supabase.table(TABLE_NAME).select(
            "id, name, ingredients, directions, ner"
        ).range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
        
        all_recipes.extend(response.data)
        offset += batch_size
        print(f"   Fetched {len(all_recipes)} recipes...")
        
        if limit and len(all_recipes) >= limit:
            all_recipes = all_recipes[:limit]
            break
        
        if len(response.data) < batch_size:
            break
    
    print(f"✅ Fetched {len(all_recipes)} recipes total")
    return all_recipes


def update_recipe_batch(supabase: Client, updates: list[dict]) -> int:
    """Update recipes in batch."""
    success = 0
    for update in updates:
        try:
            supabase.table(TABLE_NAME).update({
                "graph_representation": update["graph"]
            }).eq("id", update["id"]).execute()
            success += 1
        except Exception as e:
            print(f"    Error updating recipe {update['id']}: {e}")
    return success


def generate_all_graphs(limit: int = None, batch_size: int = 100):
    """Main function to generate graph representations for all recipes."""
    print("=" * 60)
    print("REGEX-BASED GRAPH REPRESENTATION GENERATOR")
    print(f"Target table: {TABLE_NAME}")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    # Fetch recipes
    recipes = fetch_recipes(supabase, limit=limit)
    
    if not recipes:
        print("✅ No recipes to process!")
        return
    
    print(f"\n🔄 Generating graphs for {len(recipes)} recipes...")
    
    updates = []
    success_count = 0
    fallback_count = 0
    
    for recipe in tqdm(recipes, desc="Processing"):
        recipe_id = recipe["id"]
        recipe_name = recipe.get("name", "Unknown")
        
        # Parse fields
        ingredients = parse_json_field(recipe.get("ingredients", []))
        directions = parse_json_field(recipe.get("directions", []))
        ner = parse_json_field(recipe.get("ner", []))
        
        # Generate graph
        graph = generate_graph_from_recipe(recipe_name, ingredients, directions, ner)
        
        if graph["metadata"].get("fallback"):
            fallback_count += 1
        else:
            success_count += 1
        
        updates.append({"id": recipe_id, "graph": graph})
        
        # Batch update
        if len(updates) >= batch_size:
            update_recipe_batch(supabase, updates)
            updates = []
    
    # Final batch
    if updates:
        update_recipe_batch(supabase, updates)
    
    print("\n" + "=" * 60)
    print("GRAPH GENERATION COMPLETE!")
    print("=" * 60)
    print(f"  ✅ Full graphs: {success_count}")
    print(f"  ⚠️  Simple/Fallback: {fallback_count}")
    print(f"  📊 Total processed: {len(recipes)}")


def verify_graphs(sample_size: int = 5):
    """Verify graph representations by showing samples."""
    print("\n" + "=" * 60)
    print(f"VERIFYING GRAPHS IN {TABLE_NAME}")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    response = supabase.table(TABLE_NAME).select(
        "id, name, ner, graph_representation"
    ).limit(sample_size).execute()
    
    for recipe in response.data:
        print(f"\n📌 {recipe['name']}")
        
        ner = parse_json_field(recipe.get('ner', []))
        print(f"   Ingredients: {len(ner)}")
        
        graph = recipe.get('graph_representation', {})
        if graph:
            print(f"   Nodes: {len(graph.get('nodes', []))}")
            print(f"   Edges: {len(graph.get('edges', []))}")
            print(f"   Method: {graph.get('metadata', {}).get('generation_method', 'unknown')}")
            
            # Show sample edges
            edges = graph.get('edges', [])[:3]
            for edge in edges:
                print(f"      Edge: {edge['source']} --[{edge['action']}]--> {edge['target']}")
        else:
            print("   ❌ No graph representation")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verify":
            sample = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            verify_graphs(sample_size=sample)
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            generate_all_graphs(limit=limit)
        else:
            print("Usage:")
            print("  python 3.5_generate_graph_representations.py           # Process all recipes")
            print("  python 3.5_generate_graph_representations.py --limit N # Process N recipes")
            print("  python 3.5_generate_graph_representations.py --verify  # Verify existing graphs")
    else:
        generate_all_graphs()