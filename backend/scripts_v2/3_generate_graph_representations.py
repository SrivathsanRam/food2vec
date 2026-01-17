"""
Generate and validate graph representations for recipes using Gemini API.
Creates action-state graphs showing how ingredients combine through cooking steps.

Supports both OpenAI and Gemini APIs (configurable).
"""

import os
import json
import time
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Target table
TABLE_NAME = "recipes"

# Rate limiting settings
OPENAI_REQUESTS_PER_MINUTE = 60  # Adjust based on your tier
DELAY_BETWEEN_REQUESTS = 1  # seconds


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_openai():
    """Initialize OpenAI API client."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured in .env")
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        print("Install openai: pip install openai")
        raise


def create_graph_prompt(recipe_name: str, ingredients: list, directions: list, ner: list) -> str:
    """Create a prompt for the LLM to generate the graph representation."""
    
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


def call_openai(client, prompt: str) -> str | None:
    """Call OpenAI API and return the response text."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o for better quality
            messages=[
                {"role": "system", "content": "You are a culinary AI that creates structured recipe graphs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    OpenAI API error: {e}")
        return None


def parse_llm_response(response_text: str) -> dict | None:
    """Parse and validate the LLM's JSON response."""
    if not response_text:
        return None
    
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
    """Generate a simple fallback graph if LLM fails."""
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


def fetch_recipes_needing_graphs(supabase: Client, batch_size: int = 100) -> list[dict]:
    """Fetch recipes that need graph representation."""
    print("📂 Fetching recipes needing graph representations...")
    
    all_recipes = []
    offset = 0
    
    while True:
        response = supabase.table(TABLE_NAME).select(
            "id, name, ingredients, directions, ner, graph_representation"
        ).range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
        
        # Filter recipes with empty/placeholder graphs
        for recipe in response.data:
            graph = recipe.get("graph_representation")
            needs_update = False
            
            if not graph:
                needs_update = True
            elif isinstance(graph, dict):
                metadata = graph.get("metadata", {})
                nodes = graph.get("nodes", [])
                edges = graph.get("edges", [])
                # Skip if has valid nodes and edges (already processed)
                if metadata.get("pending") or not nodes or not edges:
                    needs_update = True
                # Already has a valid graph - skip
            else:
                needs_update = True
            
            if needs_update:
                all_recipes.append(recipe)
        
        offset += batch_size
        print(f"   Checked {offset} recipes, found {len(all_recipes)} needing updates...")
        
        if len(response.data) < batch_size:
            break
    
    print(f"✅ Found {len(all_recipes)} recipes needing graph representations")
    return all_recipes


def update_recipe_graph(supabase: Client, recipe_id: int, graph: dict) -> bool:
    """Update a recipe's graph representation in the database."""
    try:
        supabase.table(TABLE_NAME).update({
            "graph_representation": graph
        }).eq("id", recipe_id).execute()
        return True
    except Exception as e:
        print(f"    ❌ Error updating recipe {recipe_id}: {e}")
        return False


def generate_graph_representations(
    limit: int = None,
    skip_existing: bool = True
):
    """Main function to generate graph representations for recipes."""
    print("=" * 60)
    print("RECIPE GRAPH REPRESENTATION GENERATOR")
    print("Using API: OpenAI")
    print("=" * 60)
    
    # Initialize clients
    supabase = get_supabase_client()
    openai_client = init_openai()
    delay = 60 / OPENAI_REQUESTS_PER_MINUTE
    
    def call_api(prompt: str) -> str | None:
        """Call OpenAI API."""
        return call_openai(openai_client, prompt)
     
    
    # Fetch recipes needing graphs
    if skip_existing:
        recipes = fetch_recipes_needing_graphs(supabase)
    else:
        # Fetch all recipes
        response = supabase.table(TABLE_NAME).select(
            "id, name, ingredients, directions, ner, graph_representation"
        ).execute()
        recipes = response.data
    
    if limit:
        recipes = recipes[:limit]
        print(f"   Limiting to {limit} recipes")
    
    if not recipes:
        print("✅ No recipes need graph representations!")
        return
    
    print(f"\n🔄 Processing {len(recipes)} recipes...")
    
    success_count = 0
    fallback_count = 0
    error_count = 0
    
    for idx, recipe in enumerate(recipes):
        recipe_id = recipe["id"]
        recipe_name = recipe["name"]
        
        print(f"\n[{idx + 1}/{len(recipes)}] {recipe_name[:50]}...")
        
        # Parse fields
        ingredients = recipe.get("ingredients", [])
        if isinstance(ingredients, str):
            try:
                ingredients = json.loads(ingredients)
            except:
                ingredients = []
        
        directions = recipe.get("directions", [])
        if isinstance(directions, str):
            try:
                directions = json.loads(directions)
            except:
                directions = []
        
        ner = recipe.get("ner", [])
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        if not ner or not directions:
            print("    ⚠️  Missing ingredients or directions, using fallback")
            graph = generate_fallback_graph(ner or [], directions or [])
            fallback_count += 1
        else:
            # Generate prompt and call API
            prompt = create_graph_prompt(recipe_name, ingredients, directions, ner)
            response_text = call_api(prompt)
            
            # Parse response
            graph = parse_llm_response(response_text)
            
            if graph:
                # Check for isolated nodes
                isolated = check_isolated_nodes(graph, ner)
                if isolated:
                    print(f"    ⚠️  {len(isolated)} isolated nodes, retrying...")
                    # Retry once
                    time.sleep(delay)
                    response_text = call_api(prompt)
                    graph = parse_llm_response(response_text)
                    
                    if not graph or check_isolated_nodes(graph, ner):
                        print("    Using fallback graph")
                        graph = generate_fallback_graph(ner, directions)
                        fallback_count += 1
                    else:
                        graph = add_metadata_to_graph(graph, ner)
                        success_count += 1
                else:
                    graph = add_metadata_to_graph(graph, ner)
                    success_count += 1
            else:
                print("    ⚠️  Failed to parse response, using fallback")
                graph = generate_fallback_graph(ner, directions)
                fallback_count += 1
        
        # Update database
        if update_recipe_graph(supabase, recipe_id, graph):
            print(f"    ✅ Updated (nodes: {len(graph['nodes'])}, edges: {len(graph['edges'])})")
        else:
            error_count += 1
        
        # Rate limiting
        time.sleep(delay)
    
    print("\n" + "=" * 60)
    print("GRAPH GENERATION COMPLETE!")
    print("=" * 60)
    print(f"  ✅ Success (API): {success_count}")
    print(f"  ⚠️  Fallback: {fallback_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"\n📌 Next step: Run 4_build_gnn_embeddings.py")


def verify_graphs(sample_size: int = 5):
    """Verify graph representations by showing samples."""
    print("\n" + "=" * 60)
    print("VERIFYING GRAPH REPRESENTATIONS")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    response = supabase.table(TABLE_NAME).select(
        "id, name, ner, graph_representation"
    ).limit(sample_size).execute()
    
    for recipe in response.data:
        print(f"\n📌 {recipe['name']}")
        
        ner = recipe.get('ner', [])
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        print(f"   Ingredients: {len(ner)}")
        
        graph = recipe.get('graph_representation', {})
        if graph:
            print(f"   Nodes: {len(graph.get('nodes', []))}")
            print(f"   Edges: {len(graph.get('edges', []))}")
            
            # Check for isolated nodes
            isolated = check_isolated_nodes(graph, ner)
            if isolated:
                print(f"   ⚠️  Isolated nodes: {isolated}")
            else:
                print("   ✅ No isolated nodes")
        else:
            print("   ❌ No graph representation")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--verify":
            verify_graphs(sample_size=10)
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            generate_graph_representations(limit=limit)
        else:
            print("Usage:")
            print("  python 3_generate_graph_representations.py           # Process all recipes needing graphs")
            print("  python 3_generate_graph_representations.py --limit N # Process N recipes")
            print("  python 3_generate_graph_representations.py --verify  # Verify existing graphs")
    else:
        generate_graph_representations()
