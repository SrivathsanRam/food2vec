"""
Cleanup script to remove recipes with zero embeddings.

Scans the database and deletes any rows where the embedding vector
is all zeros (indicating failed or incomplete embedding generation).
"""

import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target table
TABLE_NAME = "recipes"

# Batch size for fetching
BATCH_SIZE = 1000


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured. Check your .env file.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def is_zero_embedding(embedding) -> bool:
    """Check if an embedding is all zeros or invalid."""
    if embedding is None:
        return True
    
    # Parse if it's a string
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except (json.JSONDecodeError, ValueError):
            return True
    
    # Check if it's a list/array
    if not isinstance(embedding, (list, tuple)):
        return True
    
    # Check if empty
    if len(embedding) == 0:
        return True
    
    # Check if all zeros
    return all(v == 0 or v == 0.0 for v in embedding)


def fetch_all_recipes(supabase: Client) -> list[dict]:
    """Fetch all recipes with their embeddings."""
    print(f"📂 Fetching recipes from {TABLE_NAME}...")
    
    all_recipes = []
    offset = 0
    
    while True:
        response = supabase.table(TABLE_NAME).select(
            "id, name, embedding"
        ).range(offset, offset + BATCH_SIZE - 1).execute()
        
        if not response.data:
            break
        
        all_recipes.extend(response.data)
        offset += BATCH_SIZE
        print(f"   Fetched {len(all_recipes)} recipes...")
        
        if len(response.data) < BATCH_SIZE:
            break
    
    print(f"✅ Total recipes fetched: {len(all_recipes)}")
    return all_recipes


def find_zero_embedding_recipes(recipes: list[dict]) -> list[dict]:
    """Find all recipes with zero or invalid embeddings."""
    print("\n🔍 Scanning for zero embeddings...")
    
    zero_embedding_recipes = []
    
    for recipe in tqdm(recipes, desc="Checking embeddings"):
        embedding = recipe.get("embedding")
        if is_zero_embedding(embedding):
            zero_embedding_recipes.append(recipe)
    
    print(f"⚠️  Found {len(zero_embedding_recipes)} recipes with zero/invalid embeddings")
    return zero_embedding_recipes


def delete_recipes(supabase: Client, recipes: list[dict], dry_run: bool = True) -> int:
    """Delete recipes from the database."""
    if not recipes:
        print("✅ No recipes to delete!")
        return 0
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would delete {len(recipes)} recipes:")
        for recipe in recipes[:10]:  # Show first 10
            print(f"   - ID {recipe['id']}: {recipe.get('name', 'Unknown')}")
        if len(recipes) > 10:
            print(f"   ... and {len(recipes) - 10} more")
        print("\n   Run with --delete to actually remove these recipes.")
        return 0
    
    print(f"\n🗑️  Deleting {len(recipes)} recipes...")
    
    deleted = 0
    failed = 0
    
    for recipe in tqdm(recipes, desc="Deleting"):
        try:
            supabase.table(TABLE_NAME).delete().eq("id", recipe["id"]).execute()
            deleted += 1
        except Exception as e:
            print(f"   ❌ Failed to delete ID {recipe['id']}: {e}")
            failed += 1
    
    print(f"\n✅ Deleted: {deleted}")
    if failed:
        print(f"❌ Failed: {failed}")
    
    return deleted


def show_statistics(supabase: Client):
    """Show embedding statistics for the table."""
    print(f"\n📊 Embedding Statistics for {TABLE_NAME}")
    print("=" * 50)
    
    recipes = fetch_all_recipes(supabase)
    
    total = len(recipes)
    zero_count = 0
    valid_count = 0
    null_count = 0
    
    for recipe in recipes:
        embedding = recipe.get("embedding")
        if embedding is None:
            null_count += 1
        elif is_zero_embedding(embedding):
            zero_count += 1
        else:
            valid_count += 1
    
    print(f"   Total recipes: {total}")
    print(f"   Valid embeddings: {valid_count} ({100*valid_count/total:.1f}%)")
    print(f"   Zero embeddings: {zero_count} ({100*zero_count/total:.1f}%)")
    print(f"   Null embeddings: {null_count} ({100*null_count/total:.1f}%)")
    print("=" * 50)


def main():
    """Main function."""
    import sys
    
    print("=" * 60)
    print("ZERO EMBEDDING CLEANUP SCRIPT")
    print(f"Target table: {TABLE_NAME}")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            show_statistics(supabase)
            return
        elif sys.argv[1] == "--delete":
            dry_run = False
        elif sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python 6_cleanup_zero_embeddings.py           # Dry run (show what would be deleted)")
            print("  python 6_cleanup_zero_embeddings.py --delete  # Actually delete zero-embedding recipes")
            print("  python 6_cleanup_zero_embeddings.py --stats   # Show embedding statistics")
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        dry_run = True
    
    # Fetch and check recipes
    recipes = fetch_all_recipes(supabase)
    zero_recipes = find_zero_embedding_recipes(recipes)
    
    # Delete (or dry run)
    delete_recipes(supabase, zero_recipes, dry_run=dry_run)


if __name__ == "__main__":
    main()