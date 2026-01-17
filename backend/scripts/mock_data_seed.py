"""
Seed script for Supabase PostgreSQL database with pgvector extension.
Seeds recipe data from CSV with vector embeddings, ingredients, directions, and graph representations.
"""

import os
import csv
import json
import hashlib
import ast
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Vector dimension for embeddings
EMBEDDING_DIM = 512

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# CSV file path
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "strict_filtered_balanced_sample_large_6_ingredients_no_bucket.csv"


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def stable_hash(s: str) -> int:
    """Generate a stable hash for a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def generate_embedding(tokens: list[str]) -> list[float]:
    """
    Generate a deterministic mock embedding from tokens.
    In production, replace with actual embedding model (e.g., OpenAI, Sentence Transformers).
    """
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for t in tokens:
        h = stable_hash(t)
        idx = h % EMBEDDING_DIM
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        # Weight actions more heavily
        if t.startswith("ACT_"):
            weight = 2.0
        elif t.startswith("ING_"):
            weight = 1.5
        else:
            weight = 1.0
        v[idx] += sign * weight
    # L2 normalize
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v.tolist()


def extract_tokens_from_recipe(ingredients_ner: list[str], directions: list[str]) -> list[str]:
    """
    Extract tokens from recipe ingredients (NER) and directions.
    Creates ING_ tokens from ingredients and ACT_ tokens from detected actions.
    """
    tokens = []
    
    # Add ingredient tokens from NER (normalized ingredient names)
    for ing in ingredients_ner:
        # Clean and normalize ingredient name
        ing_clean = ing.strip().lower().replace(" ", "_").replace("-", "_")
        tokens.append(f"ING_{ing_clean}")
    
    # Action keywords to detect in directions
    action_keywords = [
        "mix", "chop", "dice", "slice", "sauté", "saute", "fry", "bake", 
        "boil", "simmer", "blend", "whisk", "fold", "roast", "grill", 
        "steam", "stuff", "combine", "stir", "season", "marinate", 
        "garnish", "serve", "cream", "melt", "pour", "spread", "roll",
        "cut", "add", "cook", "heat", "preheat", "drain", "cool",
        "chill", "freeze", "refrigerate", "knead", "beat", "sprinkle",
        "brush", "coat", "drop", "shape", "form", "press", "cover"
    ]
    
    # Extract actions from directions
    actions_found = set()
    for direction in directions:
        direction_lower = direction.lower()
        for action in action_keywords:
            if action in direction_lower:
                actions_found.add(f"ACT_{action}")
    
    tokens.extend(sorted(actions_found))
    
    # Extract temperature if mentioned
    for direction in directions:
        direction_lower = direction.lower()
        if "350°" in direction or "350 " in direction_lower:
            tokens.append("TEMP_350F")
        elif "375°" in direction or "375 " in direction_lower:
            tokens.append("TEMP_375F")
        elif "400°" in direction or "400 " in direction_lower:
            tokens.append("TEMP_400F")
        elif "425°" in direction or "425 " in direction_lower:
            tokens.append("TEMP_425F")
        elif "250°" in direction or "250 " in direction_lower:
            tokens.append("TEMP_250F")
        elif "325°" in direction or "325 " in direction_lower:
            tokens.append("TEMP_325F")
    
    return list(set(tokens))  # Remove duplicates


def create_graph_representation(ingredients: list[str], directions: list[str]) -> dict:
    """
    Create an action-state graph representation of a recipe.
    
    Graph structure:
    - nodes: states (ingredients, intermediate products, final dish)
    - edges: actions (cooking operations)
    """
    nodes = []
    edges = []
    node_id = 0
    
    # Add ingredient nodes (initial states)
    ingredient_nodes = {}
    for ing in ingredients:
        ingredient_nodes[ing] = node_id
        nodes.append({
            "id": node_id,
            "type": "ingredient",
            "label": ing,
            "state": "raw"
        })
        node_id += 1
    
    # Parse directions to extract actions and create state transitions
    action_keywords = {
        "mix": "mixed",
        "chop": "chopped", 
        "dice": "diced",
        "slice": "sliced",
        "sauté": "sautéed",
        "saute": "sautéed",
        "fry": "fried",
        "bake": "baked",
        "boil": "boiled",
        "simmer": "simmered",
        "blend": "blended",
        "whisk": "whisked",
        "fold": "folded",
        "roast": "roasted",
        "grill": "grilled",
        "steam": "steamed",
        "stuff": "stuffed",
        "combine": "combined",
        "stir": "stirred",
        "season": "seasoned",
        "marinate": "marinated",
        "garnish": "garnished",
        "serve": "served",
        "cream": "creamed",
        "melt": "melted",
        "pour": "poured",
        "spread": "spread",
        "roll": "rolled",
        "cut": "cut",
        "add": "added",
        "cook": "cooked",
        "heat": "heated",
        "drain": "drained",
        "cool": "cooled",
        "chill": "chilled",
        "knead": "kneaded",
        "beat": "beaten",
        "sprinkle": "sprinkled",
        "brush": "brushed",
        "drop": "dropped",
        "shape": "shaped",
        "form": "formed",
        "press": "pressed"
    }
    
    previous_state_id = None
    
    for step_idx, direction in enumerate(directions):
        direction_lower = direction.lower()
        
        # Find actions in this step
        actions_found = []
        for action, result_state in action_keywords.items():
            if action in direction_lower:
                actions_found.append((action, result_state))
        
        if actions_found:
            # Create intermediate state node
            action_names = [a[0] for a in actions_found]
            result_states = [a[1] for a in actions_found]
            
            state_node = {
                "id": node_id,
                "type": "intermediate" if step_idx < len(directions) - 1 else "final",
                "label": f"Step {step_idx + 1} result",
                "state": result_states[-1]
            }
            nodes.append(state_node)
            
            # Create edges from ingredients or previous state to new state
            if step_idx == 0:
                # First step - connect from ingredients
                for ing_id in list(ingredient_nodes.values())[:3]:  # Connect first few ingredients
                    edges.append({
                        "source": ing_id,
                        "target": node_id,
                        "action": action_names[0],
                        "step": step_idx + 1
                    })
            elif previous_state_id is not None:
                edges.append({
                    "source": previous_state_id,
                    "target": node_id,
                    "action": ", ".join(action_names),
                    "step": step_idx + 1
                })
            
            previous_state_id = node_id
            node_id += 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "ingredient_count": len(ingredients),
            "step_count": len(directions)
        }
    }


def parse_csv_field(field_str: str) -> list:
    """Parse a CSV field that contains a Python list as a string."""
    try:
        # Use ast.literal_eval to safely parse the string as a Python literal
        return ast.literal_eval(field_str)
    except (ValueError, SyntaxError):
        # If parsing fails, return empty list
        return []


def load_recipes_from_csv(csv_path: Path, limit: int = None) -> list[dict]:
    """
    Load recipes from the CSV file.
    
    CSV columns:
    - Unnamed: 0 (index)
    - title (recipe name)
    - ingredients (JSON array as string)
    - directions (JSON array as string)
    - link
    - source
    - NER (named entity recognition - simplified ingredient names)
    """
    recipes = []
    seen_names = set()  # Track seen recipe names to handle duplicates
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            
            # Parse fields
            title = row.get('title', '').strip()
            ingredients = parse_csv_field(row.get('ingredients', '[]'))
            directions = parse_csv_field(row.get('directions', '[]'))
            ner = parse_csv_field(row.get('NER', '[]'))
            link = row.get('link', '')
            source = row.get('source', '')
            
            if not title or not ingredients or not directions:
                continue
            
            # Handle duplicate recipe names by appending a suffix
            original_title = title
            suffix = 1
            while title in seen_names:
                suffix += 1
                title = f"{original_title} (v{suffix})"
            seen_names.add(title)
            
            # Generate tokens from NER (ingredient names) and directions
            tokens = extract_tokens_from_recipe(ner, directions)
            
            recipes.append({
                "name": title,
                "ingredients": ingredients,
                "directions": directions,
                "tokens": tokens,
                "link": link,
                "source": source,
                "ner": ner
            })
    
    return recipes


def create_tables_sql() -> str:
    """Generate SQL to create the recipes table with pgvector support."""
    return """
    -- Enable pgvector extension (run once)
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Create recipes table
    CREATE TABLE IF NOT EXISTS recipes (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        ingredients JSONB NOT NULL,
        directions JSONB NOT NULL,
        graph_representation JSONB NOT NULL,
        tokens JSONB NOT NULL,
        embedding vector(512) NOT NULL,
        link TEXT,
        source VARCHAR(100),
        ner JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    
    -- Create index for vector similarity search
    CREATE INDEX IF NOT EXISTS recipes_embedding_idx 
    ON recipes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    
    -- Create index on name for text search
    CREATE INDEX IF NOT EXISTS recipes_name_idx ON recipes (name);
    
    -- Create GIN index on ingredients for JSONB search
    CREATE INDEX IF NOT EXISTS recipes_ingredients_idx ON recipes USING GIN (ingredients);
    
    -- Create GIN index on NER for ingredient search
    CREATE INDEX IF NOT EXISTS recipes_ner_idx ON recipes USING GIN (ner);
    
    -- Create function for similarity search
    CREATE OR REPLACE FUNCTION search_recipes(
        query_embedding vector(512),
        match_threshold float DEFAULT 0.5,
        match_count int DEFAULT 10
    )
    RETURNS TABLE (
        id int,
        name varchar,
        ingredients jsonb,
        directions jsonb,
        graph_representation jsonb,
        similarity float
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            r.id,
            r.name,
            r.ingredients,
            r.directions,
            r.graph_representation,
            1 - (r.embedding <=> query_embedding) AS similarity
        FROM recipes r
        WHERE 1 - (r.embedding <=> query_embedding) > match_threshold
        ORDER BY r.embedding <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    
    -- Create function to search by ingredient
    CREATE OR REPLACE FUNCTION search_by_ingredient(
        ingredient_query text,
        result_limit int DEFAULT 20
    )
    RETURNS TABLE (
        id int,
        name varchar,
        ingredients jsonb,
        directions jsonb,
        ner jsonb
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT
            r.id,
            r.name,
            r.ingredients,
            r.directions,
            r.ner
        FROM recipes r
        WHERE r.ner @> to_jsonb(ARRAY[ingredient_query])
           OR EXISTS (
               SELECT 1 FROM jsonb_array_elements_text(r.ner) elem 
               WHERE elem ILIKE '%' || ingredient_query || '%'
           )
        LIMIT result_limit;
    END;
    $$;
    """


def seed_database(limit: int = None):
    """Seed the Supabase database with recipe data from CSV."""
    print("🌱 Starting database seeding...")
    
    # Check if CSV file exists
    if not CSV_FILE.exists():
        print(f"❌ Error: CSV file not found at {CSV_FILE}")
        return
    
    try:
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set the following environment variables:")
        print("  SUPABASE_URL=your_supabase_project_url")
        print("  SUPABASE_KEY=your_supabase_anon_key")
        return
    
    # Load recipes from CSV
    print(f"\n📂 Loading recipes from CSV: {CSV_FILE.name}")
    recipes = load_recipes_from_csv(CSV_FILE, limit=limit)
    print(f"✅ Loaded {len(recipes)} recipes from CSV")
    
    print(f"\n📝 Preparing recipes for insertion...")
    
    recipes_to_insert = []
    
    for idx, recipe in enumerate(recipes):
        if idx % 50 == 0:
            print(f"  Processing recipe {idx + 1}/{len(recipes)}: {recipe['name'][:50]}...")
        
        # Generate graph representation
        graph = create_graph_representation(
            recipe["ingredients"], 
            recipe["directions"]
        )
        
        # Generate embedding from tokens
        embedding = generate_embedding(recipe["tokens"])
        
        # Prepare record for insertion
        record = {
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
            "graph_representation": graph,
            "tokens": recipe["tokens"],
            "embedding": embedding,
            "link": recipe.get("link", ""),
            "source": recipe.get("source", ""),
            "ner": recipe.get("ner", [])
        }
        
        recipes_to_insert.append(record)
    
    print(f"\n📤 Inserting {len(recipes_to_insert)} recipes into database...")
    
    try:
        # Insert in batches to avoid timeout
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(recipes_to_insert), batch_size):
            batch = recipes_to_insert[i:i + batch_size]
            result = supabase.table("recipes").upsert(
                batch,
                on_conflict="name"  # Update if recipe name already exists
            ).execute()
            total_inserted += len(batch)
            print(f"  Inserted batch {i // batch_size + 1}: {total_inserted}/{len(recipes_to_insert)} recipes")
        
        print(f"\n✅ Successfully inserted/updated {total_inserted} recipes!")
        
    except Exception as e:
        print(f"❌ Error inserting recipes: {e}")
        print("\n⚠️  Make sure you have created the recipes table.")
        print("Run the following SQL in your Supabase SQL Editor:\n")
        print(create_tables_sql())
        return
    
    # Test query
    print("\n🔍 Testing similarity search...")
    test_query_tokens = ["ING_sugar", "ING_butter", "ACT_bake"]
    test_embedding = generate_embedding(test_query_tokens)
    
    try:
        # Using RPC to call the search function
        results = supabase.rpc(
            "search_recipes",
            {
                "query_embedding": test_embedding,
                "match_threshold": 0.3,
                "match_count": 3
            }
        ).execute()
        
        print("Top 3 similar recipes:")
        for r in results.data:
            print(f"  - {r['name']} (similarity: {r['similarity']:.3f})")
            
    except Exception as e:
        print(f"⚠️  Could not test search function: {e}")
        print("   This is okay if the function hasn't been created yet.")
    
    print("\n✨ Database seeding complete!")


def print_setup_instructions():
    """Print setup instructions for the user."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Supabase + pgvector Setup Instructions             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Create a Supabase project at https://supabase.com        ║
║                                                              ║
║  2. In your Supabase SQL Editor, run the following SQL:      ║
║     (This enables pgvector and creates the recipes table)    ║
║                                                              ║
║  3. Set environment variables in your .env file:             ║
║     SUPABASE_URL=https://your-project.supabase.co           ║
║     SUPABASE_KEY=your-anon-key                              ║
║                                                              ║
║  4. Run this script: python mock_data_seed.py               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("\nSQL to create tables:\n")
    print(create_tables_sql())


def preview_csv_data(num_rows: int = 5):
    """Preview the first few rows of the CSV file."""
    if not CSV_FILE.exists():
        print(f"❌ Error: CSV file not found at {CSV_FILE}")
        return
    
    recipes = load_recipes_from_csv(CSV_FILE, limit=num_rows)
    
    print(f"\n📋 Preview of first {num_rows} recipes:\n")
    for idx, recipe in enumerate(recipes):
        print(f"{'='*60}")
        print(f"Recipe {idx + 1}: {recipe['name']}")
        print(f"{'='*60}")
        print(f"Ingredients ({len(recipe['ingredients'])}):")
        for ing in recipe['ingredients'][:5]:
            print(f"  - {ing}")
        if len(recipe['ingredients']) > 5:
            print(f"  ... and {len(recipe['ingredients']) - 5} more")
        
        print(f"\nDirections ({len(recipe['directions'])} steps):")
        for i, step in enumerate(recipe['directions'][:3]):
            print(f"  {i+1}. {step[:80]}...")
        if len(recipe['directions']) > 3:
            print(f"  ... and {len(recipe['directions']) - 3} more steps")
        
        print(f"\nTokens ({len(recipe['tokens'])}): {', '.join(recipe['tokens'][:10])}")
        if len(recipe['tokens']) > 10:
            print(f"  ... and {len(recipe['tokens']) - 10} more tokens")
        
        print(f"\nNER: {recipe['ner']}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--setup":
            print_setup_instructions()
        elif sys.argv[1] == "--preview":
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            preview_csv_data(num)
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            seed_database(limit=limit)
        else:
            print("Usage:")
            print("  python mock_data_seed.py           # Seed all recipes from CSV")
            print("  python mock_data_seed.py --setup   # Show setup instructions")
            print("  python mock_data_seed.py --preview [n]  # Preview first n recipes")
            print("  python mock_data_seed.py --limit [n]    # Seed only first n recipes")
    else:
        seed_database()
