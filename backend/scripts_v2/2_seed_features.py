"""
Seed recipe features into Supabase PostgreSQL database.
Seeds: name, ingredients, directions, NER, link, source, created_at
Does NOT generate embeddings or graph representations (handled by other scripts).
"""

import os
import csv
import ast
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target table
TABLE_NAME = "recipes_duplicate"

# CSV file path
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "strict_filtered_balanced_sample_large_6_ingredients_no_bucket.csv"


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_KEY in backend/.env"
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_csv_field(field_str: str) -> list:
    """Parse a CSV field that contains a Python list as a string."""
    if not field_str:
        return []
    try:
        return ast.literal_eval(field_str)
    except (ValueError, SyntaxError):
        return []


def extract_tokens_from_recipe(ingredients_ner: list[str], directions: list[str]) -> list[str]:
    """
    Extract tokens from recipe ingredients (NER) and directions.
    Creates ING_ tokens from ingredients and ACT_ tokens from detected actions.
    """
    tokens = []
    
    # Add ingredient tokens from NER (normalized ingredient names)
    for ing in ingredients_ner:
        clean_ing = ing.strip().lower()
        if clean_ing:
            tokens.append(f"ING_{clean_ing.replace(' ', '_')}")
    
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
    
    return list(set(tokens))  # Remove duplicates


def load_recipes_from_csv(csv_path: Path, limit: int = None) -> list[dict]:
    """
    Load recipes from the CSV file.
    
    CSV columns:
    - title (recipe name)
    - ingredients (JSON array as string)
    - directions (JSON array as string)
    - link
    - source
    - NER (named entity recognition - simplified ingredient names)
    """
    recipes = []
    seen_names = set()  # Track seen recipe names to handle duplicates
    
    print(f"📂 Loading recipes from: {csv_path.name}")
    
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
            
            if (idx + 1) % 1000 == 0:
                print(f"   Loaded {idx + 1} recipes...")
    
    print(f"✅ Loaded {len(recipes)} recipes")
    return recipes


def create_tables_sql() -> str:
    """Generate SQL to create the recipes table with pgvector support."""
    return """
-- Enable pgvector extension (run once in Supabase SQL editor)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create recipes table
CREATE TABLE IF NOT EXISTS recipes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    ingredients JSONB NOT NULL,
    directions JSONB NOT NULL,
    graph_representation JSONB,
    tokens JSONB NOT NULL,
    embedding vector(512),
    link TEXT,
    source VARCHAR(100),
    ner JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for vector similarity search (run after seeding embeddings)
-- CREATE INDEX IF NOT EXISTS recipes_embedding_idx 
-- ON recipes USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Create index on name for text search
CREATE INDEX IF NOT EXISTS recipes_name_idx ON recipes (name);

-- Create GIN index on ingredients for JSONB search
CREATE INDEX IF NOT EXISTS recipes_ingredients_idx ON recipes USING GIN (ingredients);

-- Create GIN index on NER for ingredient search
CREATE INDEX IF NOT EXISTS recipes_ner_idx ON recipes USING GIN (ner);
"""


def seed_database(limit: int = None, batch_size: int = 100):
    """Seed the Supabase database with recipe data from CSV."""
    print("=" * 60)
    print("RECIPE DATABASE SEEDER")
    print("Seeds: name, ingredients, directions, NER, tokens")
    print("=" * 60)
    
    # Check if CSV file exists
    if not CSV_FILE.exists():
        print(f"\n❌ CSV file not found: {CSV_FILE}")
        print("   Run 1_download_kaggle_data.py first to download and filter the data.")
        return
    
    # Initialize Supabase client
    try:
        supabase = get_supabase_client()
        print("✅ Connected to Supabase")
    except ValueError as e:
        print(f"\n❌ {e}")
        print_setup_instructions()
        return
    
    # Load recipes from CSV
    recipes = load_recipes_from_csv(CSV_FILE, limit=limit)
    
    if not recipes:
        print("❌ No recipes loaded from CSV")
        return
    
    # Prepare recipes for insertion
    print(f"\n📝 Preparing {len(recipes)} recipes for insertion...")
    
    recipes_to_insert = []
    
    for recipe in recipes:
        # Create placeholder embedding (will be updated by embedding script)
        placeholder_embedding = [0.0] * 512
        
        # Create empty graph representation (will be updated by graph script)
        empty_graph = {
            "nodes": [],
            "edges": [],
            "metadata": {"pending": True}
        }
        
        recipes_to_insert.append({
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
            "graph_representation": empty_graph,
            "tokens": recipe["tokens"],
            "embedding": placeholder_embedding,
            "link": recipe.get("link", ""),
            "source": recipe.get("source", ""),
            "ner": recipe["ner"]
        })
    
    # Insert in batches
    print(f"\n📤 Inserting {len(recipes_to_insert)} recipes into database...")
    
    inserted_count = 0
    error_count = 0
    
    for i in range(0, len(recipes_to_insert), batch_size):
        batch = recipes_to_insert[i:i + batch_size]
        
        try:
            # Use upsert to handle duplicates gracefully
            response = supabase.table(TABLE_NAME).upsert(
                batch,
                on_conflict="name"
            ).execute()
            
            inserted_count += len(batch)
            print(f"   Inserted {inserted_count}/{len(recipes_to_insert)} recipes...")
            
        except Exception as e:
            error_str = str(e)
            
            # If bulk insert fails, try individual inserts
            if "duplicate key" in error_str.lower() or "unique constraint" in error_str.lower():
                for recipe in batch:
                    try:
                        supabase.table(TABLE_NAME).upsert(
                            recipe,
                            on_conflict="name"
                        ).execute()
                        inserted_count += 1
                    except Exception as inner_e:
                        error_count += 1
            else:
                print(f"   ❌ Batch error: {error_str[:100]}")
                error_count += len(batch)
    
    print("\n" + "=" * 60)
    print("SEEDING COMPLETE!")
    print("=" * 60)
    print(f"  ✅ Inserted/Updated: {inserted_count}")
    print(f"  ❌ Errors: {error_count}")
    print("\n📌 Next steps:")
    print("   1. Run 3_generate_graph_representations.py to create graph representations")
    print("   2. Run 4_build_gnn_embeddings.py to create embeddings")
    print("   3. Run 5_validate_embeddings.py to validate the results")


def print_setup_instructions():
    """Print setup instructions for Supabase."""
    print("\n" + "=" * 60)
    print("SUPABASE SETUP INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Create a Supabase project at https://supabase.com

2. Enable pgvector extension:
   - Go to SQL Editor
   - Run: CREATE EXTENSION IF NOT EXISTS vector;

3. Create the recipes table:
   - Go to SQL Editor
   - Run the SQL from create_tables_sql() function

4. Configure environment variables in backend/.env:
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_key

5. Run this script again
""")


def check_database_status():
    """Check the current status of the database."""
    print("\n" + "=" * 60)
    print("DATABASE STATUS CHECK")
    print("=" * 60)
    
    try:
        supabase = get_supabase_client()
        
        # Count recipes
        response = supabase.table(TABLE_NAME).select("id", count="exact").execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        
        print(f"\n📊 Total recipes in database: {count}")
        
        # Sample recipes
        sample = supabase.table(TABLE_NAME).select(
            "id, name, ner"
        ).limit(5).execute()
        
        if sample.data:
            print("\nSample recipes:")
            for recipe in sample.data:
                ner = recipe.get('ner', [])
                ing_count = len(ner) if isinstance(ner, list) else 0
                print(f"  [{recipe['id']}] {recipe['name']} ({ing_count} ingredients)")
        
        # Check for missing data
        missing_graphs = supabase.table(TABLE_NAME).select(
            "id", count="exact"
        ).is_("graph_representation", "null").execute()
        
        missing_embeddings = supabase.table(TABLE_NAME).select(
            "id", count="exact"
        ).is_("embedding", "null").execute()
        
        print(f"\n⚠️  Recipes missing graph_representation: {missing_graphs.count if hasattr(missing_graphs, 'count') else '?'}")
        print(f"⚠️  Recipes missing embedding: {missing_embeddings.count if hasattr(missing_embeddings, 'count') else '?'}")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            check_database_status()
        elif sys.argv[1] == "--sql":
            print(create_tables_sql())
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            seed_database(limit=limit)
        else:
            print("Usage:")
            print("  python 2_seed_features.py          # Seed all recipes")
            print("  python 2_seed_features.py --limit N # Seed first N recipes")
            print("  python 2_seed_features.py --status  # Check database status")
            print("  python 2_seed_features.py --sql     # Print SQL setup commands")
    else:
        seed_database()
