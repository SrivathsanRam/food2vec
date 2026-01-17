"""
Fix recipe graph representations using Gemini API.
Ensures no ingredient is an isolated node by analyzing directions
to accurately model ingredient relationships and cooking actions.
"""

import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Rate limiting settings
REQUESTS_PER_MINUTE = 15  # Gemini free tier limit
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_gemini():
    """Initialize Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-1.5-flash')


def create_graph_prompt(recipe_name: str, ingredients: list, directions: list, ner: list) -> str:
    """Create a prompt for Gemini to generate the graph representation."""
    
    prompt = f"""Analyze this recipe and create a detailed cooking action graph.

RECIPE: {recipe_name}

INGREDIENTS:
{json.dumps(ingredients, indent=2)}

INGREDIENT NAMES (NER):
{json.dumps(ner, indent=2)}

DIRECTIONS:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(directions)])}

Create a JSON graph representation with this EXACT structure:
{{
    "nodes": [
        {{"id": 0, "type": "ingredient", "label": "ingredient_name", "state": "raw"}},
        {{"id": 1, "type": "intermediate", "label": "description", "state": "state_after_action"}},
        {{"id": 2, "type": "final", "label": "final_dish", "state": "complete"}}
    ],
    "edges": [
        {{"source": 0, "target": 1, "action": "action_verb", "step": 1}}
    ]
}}

RULES:
1. EVERY ingredient from NER must appear as a node with type "ingredient"
2. NO ingredient can be isolated - each must connect to at least one other node
3. Create intermediate nodes for combined ingredients or cooking stages
4. Actions should be specific verbs: mix, chop, sauté, bake, fold, cream, melt, etc.
5. The final node should represent the completed dish
6. Follow the actual sequence of directions
7. If multiple ingredients are combined in one step, connect them all to the same intermediate node
8. Node IDs must be sequential integers starting from 0

Return ONLY valid JSON, no markdown or explanation."""

    return prompt


def parse_gemini_response(response_text: str) -> dict | None:
    """Parse and validate Gemini's JSON response."""
    try:
        # Clean up the response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Parse JSON
        graph = json.loads(text)
        
        # Validate structure
        if "nodes" not in graph or "edges" not in graph:
            print("    Warning: Missing nodes or edges in response")
            return None
        
        if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
            print("    Warning: Invalid nodes or edges format")
            return None
        
        # Validate nodes
        node_ids = set()
        for node in graph["nodes"]:
            if not all(k in node for k in ["id", "type", "label"]):
                print("    Warning: Invalid node structure")
                return None
            node_ids.add(node["id"])
        
        # Validate edges reference existing nodes
        for edge in graph["edges"]:
            if not all(k in edge for k in ["source", "target", "action"]):
                print("    Warning: Invalid edge structure")
                return None
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                print(f"    Warning: Edge references non-existent node")
                return None
        
        return graph
        
    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse error: {e}")
        return None


def check_isolated_nodes(graph: dict, ner: list) -> list[str]:
    """Check for isolated ingredient nodes and return their labels."""
    if not graph or "nodes" not in graph or "edges" not in graph:
        return ner  # All ingredients are isolated if no graph
    
    # Get all node IDs that have edges
    connected_nodes = set()
    for edge in graph["edges"]:
        connected_nodes.add(edge["source"])
        connected_nodes.add(edge["target"])
    
    # Find ingredient nodes that are not connected
    isolated = []
    for node in graph["nodes"]:
        if node.get("type") == "ingredient":
            if node["id"] not in connected_nodes:
                isolated.append(node.get("label", "unknown"))
    
    return isolated


def add_metadata_to_graph(graph: dict, ner: list) -> dict:
    """Add metadata to the graph representation."""
    ingredient_count = len([n for n in graph["nodes"] if n.get("type") == "ingredient"])
    intermediate_count = len([n for n in graph["nodes"] if n.get("type") == "intermediate"])
    final_count = len([n for n in graph["nodes"] if n.get("type") == "final"])
    
    graph["metadata"] = {
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
        "ingredient_count": ingredient_count,
        "intermediate_count": intermediate_count,
        "final_count": final_count,
        "original_ner_count": len(ner),
        "step_count": max([e.get("step", 0) for e in graph["edges"]] + [0])
    }
    
    return graph


def generate_fallback_graph(ner: list, directions: list) -> dict:
    """Generate a simple fallback graph if Gemini fails."""
    nodes = []
    edges = []
    
    # Add ingredient nodes
    for idx, ing in enumerate(ner):
        nodes.append({
            "id": idx,
            "type": "ingredient",
            "label": ing,
            "state": "raw"
        })
    
    # Add intermediate nodes based on direction count
    n_steps = min(len(directions), 5)  # Limit intermediate nodes
    ingredient_count = len(ner)
    
    # Create intermediate nodes
    for step in range(n_steps):
        node_id = ingredient_count + step
        nodes.append({
            "id": node_id,
            "type": "intermediate" if step < n_steps - 1 else "final",
            "label": f"Step {step + 1}",
            "state": "processed"
        })
        
        if step == 0:
            # Connect all ingredients to first intermediate node
            for ing_idx in range(ingredient_count):
                edges.append({
                    "source": ing_idx,
                    "target": node_id,
                    "action": "combine",
                    "step": step + 1
                })
        else:
            # Connect previous intermediate to this one
            edges.append({
                "source": node_id - 1,
                "target": node_id,
                "action": "process",
                "step": step + 1
            })
    
    # If no directions, just connect all ingredients to a final node
    if n_steps == 0:
        final_id = ingredient_count
        nodes.append({
            "id": final_id,
            "type": "final",
            "label": "Final Dish",
            "state": "complete"
        })
        for ing_idx in range(ingredient_count):
            edges.append({
                "source": ing_idx,
                "target": final_id,
                "action": "combine",
                "step": 1
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "ingredient_count": ingredient_count,
            "step_count": n_steps,
            "fallback": True
        }
    }


def fetch_recipes_needing_fix(supabase: Client, batch_size: int = 100) -> list[dict]:
    """Fetch recipes that need graph representation fixes."""
    print("Fetching recipes from database...")
    
    all_recipes = []
    offset = 0
    
    while True:
        response = supabase.table("recipes").select(
            "id, name, ingredients, directions, ner, graph_representation"
        ).range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
        
        # Filter recipes with isolated nodes or missing graphs
        for recipe in response.data:
            ner = recipe.get("ner") or []
            if isinstance(ner, str):
                try:
                    ner = json.loads(ner)
                except:
                    ner = []
            
            graph = recipe.get("graph_representation")
            if isinstance(graph, str):
                try:
                    graph = json.loads(graph)
                except:
                    graph = None
            
            isolated = check_isolated_nodes(graph, ner)
            
            if isolated or not graph:
                recipe["_isolated_ingredients"] = isolated
                recipe["_ner_parsed"] = ner
                all_recipes.append(recipe)
        
        offset += batch_size
        print(f"  Checked {offset} recipes, found {len(all_recipes)} needing fixes...")
        
        if len(response.data) < batch_size:
            break
    
    print(f"Total recipes needing fixes: {len(all_recipes)}")
    return all_recipes


def update_recipe_graph(supabase: Client, recipe_id: int, graph: dict) -> bool:
    """Update a recipe's graph representation in the database."""
    try:
        supabase.table("recipes").update({
            "graph_representation": graph
        }).eq("id", recipe_id).execute()
        return True
    except Exception as e:
        print(f"    Error updating recipe {recipe_id}: {e}")
        return False


def fix_recipe_graphs(limit: int = None, skip_existing: bool = True):
    """Main function to fix recipe graph representations."""
    print("=" * 60)
    print("RECIPE GRAPH REPRESENTATION FIXER")
    print("Using Gemini API for intelligent graph generation")
    print("=" * 60)
    
    # Initialize clients
    supabase = get_supabase_client()
    model = init_gemini()
    
    # Fetch recipes needing fixes
    recipes = fetch_recipes_needing_fix(supabase)
    
    if limit:
        recipes = recipes[:limit]
        print(f"Processing limited to {limit} recipes")
    
    if not recipes:
        print("No recipes need fixing!")
        return
    
    print(f"\nProcessing {len(recipes)} recipes...")
    
    fixed_count = 0
    failed_count = 0
    fallback_count = 0
    
    for idx, recipe in enumerate(recipes):
        recipe_id = recipe["id"]
        recipe_name = recipe["name"]
        
        print(f"\n[{idx + 1}/{len(recipes)}] Processing: {recipe_name[:50]}...")
        
        # Parse fields
        ingredients = recipe.get("ingredients") or []
        if isinstance(ingredients, str):
            try:
                ingredients = json.loads(ingredients)
            except:
                ingredients = []
        
        directions = recipe.get("directions") or []
        if isinstance(directions, str):
            try:
                directions = json.loads(directions)
            except:
                directions = []
        
        ner = recipe.get("_ner_parsed") or []
        isolated = recipe.get("_isolated_ingredients") or []
        
        print(f"  Ingredients: {len(ner)}, Isolated: {len(isolated)}")
        
        # Generate prompt and call Gemini
        prompt = create_graph_prompt(recipe_name, ingredients, directions, ner)
        
        try:
            response = model.generate_content(prompt)
            graph = parse_gemini_response(response.text)
            
            if graph:
                # Verify no isolated nodes
                remaining_isolated = check_isolated_nodes(graph, ner)
                
                if remaining_isolated:
                    print(f"  Warning: Still has {len(remaining_isolated)} isolated nodes, using fallback")
                    graph = generate_fallback_graph(ner, directions)
                    fallback_count += 1
                else:
                    print(f"  Generated graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
                
                # Add metadata
                graph = add_metadata_to_graph(graph, ner)
                
                # Update database
                if update_recipe_graph(supabase, recipe_id, graph):
                    fixed_count += 1
                    print(f"  ✓ Updated successfully")
                else:
                    failed_count += 1
            else:
                print(f"  Using fallback graph (Gemini response invalid)")
                graph = generate_fallback_graph(ner, directions)
                graph = add_metadata_to_graph(graph, ner)
                
                if update_recipe_graph(supabase, recipe_id, graph):
                    fixed_count += 1
                    fallback_count += 1
                    print(f"  ✓ Updated with fallback")
                else:
                    failed_count += 1
                    
        except Exception as e:
            print(f"  Error calling Gemini: {e}")
            
            # Use fallback
            graph = generate_fallback_graph(ner, directions)
            graph = add_metadata_to_graph(graph, ner)
            
            if update_recipe_graph(supabase, recipe_id, graph):
                fixed_count += 1
                fallback_count += 1
                print(f"  ✓ Updated with fallback")
            else:
                failed_count += 1
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print("\n" + "=" * 60)
    print("GRAPH FIXING COMPLETE!")
    print("=" * 60)
    print(f"  Total processed: {len(recipes)}")
    print(f"  Successfully fixed: {fixed_count}")
    print(f"  Used fallback: {fallback_count}")
    print(f"  Failed: {failed_count}")


def verify_graphs(sample_size: int = 5):
    """Verify graph representations by showing samples."""
    print("\n" + "=" * 60)
    print("VERIFYING GRAPH REPRESENTATIONS")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    response = supabase.table("recipes").select(
        "id, name, ner, graph_representation"
    ).limit(sample_size).execute()
    
    for recipe in response.data:
        print(f"\n{'='*50}")
        print(f"Recipe: {recipe['name']}")
        
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        graph = recipe.get("graph_representation")
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except:
                graph = None
        
        if graph:
            print(f"Ingredients (NER): {ner}")
            print(f"Nodes: {len(graph.get('nodes', []))}")
            print(f"Edges: {len(graph.get('edges', []))}")
            
            isolated = check_isolated_nodes(graph, ner)
            if isolated:
                print(f"⚠️  Isolated ingredients: {isolated}")
            else:
                print("✓ No isolated ingredients")
            
            # Show graph structure
            print("\nGraph structure:")
            for node in graph.get("nodes", [])[:5]:
                print(f"  Node {node['id']}: {node['type']} - {node['label']}")
            if len(graph.get("nodes", [])) > 5:
                print(f"  ... and {len(graph['nodes']) - 5} more nodes")
            
            print("\nEdges:")
            for edge in graph.get("edges", [])[:5]:
                print(f"  {edge['source']} --[{edge['action']}]--> {edge['target']}")
            if len(graph.get("edges", [])) > 5:
                print(f"  ... and {len(graph['edges']) - 5} more edges")
        else:
            print("  No graph representation!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verify":
            sample = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            verify_graphs(sample)
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            fix_recipe_graphs(limit=limit)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python fix_graph_representations.py           # Fix all recipes")
            print("  python fix_graph_representations.py --limit N # Fix only N recipes")
            print("  python fix_graph_representations.py --verify  # Verify graph quality")
        else:
            fix_recipe_graphs()
    else:
        fix_recipe_graphs()