"""
Build graph-based embeddings for recipes using ingredient and action relationships.
This creates meaningful vector representations where similar recipes are close in vector space.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

EMBEDDING_DIM = 512


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_recipes(supabase: Client) -> list[dict]:
    """Fetch all recipes from the database."""
    print("Fetching recipes from database...")
    
    all_recipes = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("recipes").select(
            "id, name, ingredients, ner, tokens"
        ).range(offset, offset + page_size - 1).execute()
        
        if not response.data:
            break
            
        all_recipes.extend(response.data)
        offset += page_size
        print(f"  Fetched {len(all_recipes)} recipes...")
        
        if len(response.data) < page_size:
            break
    
    print(f"Total recipes: {len(all_recipes)}")
    return all_recipes


def build_vocabulary(recipes: list[dict]) -> tuple[dict, dict, dict, dict]:
    """
    Build vocabularies for ingredients and actions.
    Returns mappings from items to indices and vice versa.
    """
    print("Building vocabulary...")
    
    ingredient_counts = defaultdict(int)
    action_counts = defaultdict(int)
    
    for recipe in recipes:
        # ner contains normalized ingredient names (lowercase column name)
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        for ing in ner:
            if ing:
                ingredient_counts[ing.lower().strip()] += 1
        
        # Extract actions from tokens field (ACT_ prefixed)
        tokens = recipe.get("tokens") or []
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except:
                tokens = []
        
        for token in tokens:
            if token and token.startswith("ACT_"):
                action = token[4:].lower().strip()  # Remove ACT_ prefix
                action_counts[action] += 1
    
    # Filter to items that appear at least twice (reduces noise)
    min_count = 2
    ingredients = [ing for ing, count in ingredient_counts.items() if count >= min_count]
    actions = [act for act, count in action_counts.items() if count >= min_count]
    
    # Create mappings
    ing_to_idx = {ing: idx for idx, ing in enumerate(ingredients)}
    idx_to_ing = {idx: ing for ing, idx in ing_to_idx.items()}
    
    act_to_idx = {act: idx for idx, act in enumerate(actions)}
    idx_to_act = {idx: act for act, idx in act_to_idx.items()}
    
    print(f"  Unique ingredients: {len(ingredients)}")
    print(f"  Unique actions: {len(actions)}")
    
    return ing_to_idx, idx_to_ing, act_to_idx, idx_to_act


def build_cooccurrence_matrix(
    recipes: list[dict],
    ing_to_idx: dict,
    act_to_idx: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build co-occurrence matrices:
    1. Ingredient-Ingredient: How often ingredients appear together
    2. Recipe-Feature: Each recipe's features (ingredients + actions)
    """
    print("Building co-occurrence matrices...")
    
    n_ingredients = len(ing_to_idx)
    n_actions = len(act_to_idx)
    n_recipes = len(recipes)
    n_features = n_ingredients + n_actions
    
    # Ingredient co-occurrence matrix
    ing_cooccur = np.zeros((n_ingredients, n_ingredients), dtype=np.float32)
    
    # Recipe-feature matrix (sparse)
    recipe_features = np.zeros((n_recipes, n_features), dtype=np.float32)
    
    for recipe_idx, recipe in enumerate(recipes):
        # Get ingredients from ner field (lowercase)
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        ing_indices = []
        for ing in ner:
            if ing:
                ing_lower = ing.lower().strip()
                if ing_lower in ing_to_idx:
                    idx = ing_to_idx[ing_lower]
                    ing_indices.append(idx)
                    recipe_features[recipe_idx, idx] = 1.0
        
        # Update ingredient co-occurrence
        for i in range(len(ing_indices)):
            for j in range(i + 1, len(ing_indices)):
                ing_cooccur[ing_indices[i], ing_indices[j]] += 1
                ing_cooccur[ing_indices[j], ing_indices[i]] += 1
        
        # Get actions from tokens field (ACT_ prefixed)
        tokens = recipe.get("tokens") or []
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except:
                tokens = []
        
        for token in tokens:
            if token and token.startswith("ACT_"):
                action_lower = token[4:].lower().strip()
                if action_lower in act_to_idx:
                    idx = n_ingredients + act_to_idx[action_lower]
                    recipe_features[recipe_idx, idx] = 1.0
    
    print(f"  Ingredient co-occurrence matrix: {ing_cooccur.shape}")
    print(f"  Recipe-feature matrix: {recipe_features.shape}")
    
    return ing_cooccur, recipe_features


def compute_ingredient_embeddings(
    ing_cooccur: np.ndarray,
    embedding_dim: int = 128
) -> np.ndarray:
    """
    Compute ingredient embeddings using SVD on the co-occurrence matrix.
    This is similar to how Word2Vec works conceptually.
    """
    print("Computing ingredient embeddings via SVD...")
    
    # Apply PPMI (Positive Pointwise Mutual Information) transformation
    row_sums = ing_cooccur.sum(axis=1, keepdims=True) + 1e-10
    col_sums = ing_cooccur.sum(axis=0, keepdims=True) + 1e-10
    total = ing_cooccur.sum() + 1e-10
    
    # PMI = log(P(i,j) / (P(i) * P(j)))
    pmi = np.log((ing_cooccur * total) / (row_sums * col_sums) + 1e-10)
    ppmi = np.maximum(pmi, 0)  # Positive PMI only
    
    # SVD decomposition
    k = min(embedding_dim, ppmi.shape[0] - 1, ppmi.shape[1] - 1)
    if k < 1:
        print("  Warning: Not enough data for SVD, using random embeddings")
        return np.random.randn(ing_cooccur.shape[0], embedding_dim).astype(np.float32)
    
    sparse_ppmi = csr_matrix(ppmi)
    U, S, Vt = svds(sparse_ppmi, k=k)
    
    # Ingredient embeddings = U * sqrt(S)
    ing_embeddings = U * np.sqrt(S)
    
    # Pad or truncate to target dimension
    if ing_embeddings.shape[1] < embedding_dim:
        padding = np.zeros((ing_embeddings.shape[0], embedding_dim - ing_embeddings.shape[1]))
        ing_embeddings = np.hstack([ing_embeddings, padding])
    else:
        ing_embeddings = ing_embeddings[:, :embedding_dim]
    
    print(f"  Ingredient embeddings shape: {ing_embeddings.shape}")
    return ing_embeddings.astype(np.float32)


def compute_recipe_embeddings(
    recipes: list[dict],
    ing_to_idx: dict,
    act_to_idx: dict,
    ing_embeddings: np.ndarray,
    embedding_dim: int = 512
) -> dict[int, list[float]]:
    """
    Compute recipe embeddings by aggregating ingredient embeddings.
    Uses weighted average with TF-IDF-like weighting.
    """
    print("Computing recipe embeddings...")
    
    # Compute IDF for ingredients (rarer ingredients are more important)
    n_recipes = len(recipes)
    ing_doc_freq = defaultdict(int)
    
    for recipe in recipes:
        # Use ner field (lowercase)
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        seen = set()
        for ing in ner:
            if ing:
                ing_lower = ing.lower().strip()
                if ing_lower not in seen and ing_lower in ing_to_idx:
                    ing_doc_freq[ing_lower] += 1
                    seen.add(ing_lower)
    
    # IDF = log(N / df)
    idf = {}
    for ing, df in ing_doc_freq.items():
        idf[ing] = np.log(n_recipes / (df + 1))
    
    # Compute embeddings for each recipe
    recipe_embeddings = {}
    ing_embed_dim = ing_embeddings.shape[1]
    
    for recipe in recipes:
        recipe_id = recipe["id"]
        
        # Get ingredients from ner field (lowercase)
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        # Aggregate ingredient embeddings with IDF weighting
        weighted_sum = np.zeros(ing_embed_dim, dtype=np.float32)
        total_weight = 0.0
        
        for ing in ner:
            if ing:
                ing_lower = ing.lower().strip()
                if ing_lower in ing_to_idx:
                    idx = ing_to_idx[ing_lower]
                    weight = idf.get(ing_lower, 1.0)
                    weighted_sum += weight * ing_embeddings[idx]
                    total_weight += weight
        
        if total_weight > 0:
            recipe_vec = weighted_sum / total_weight
        else:
            recipe_vec = np.zeros(ing_embed_dim, dtype=np.float32)
        
        # Pad or truncate to target dimension
        if len(recipe_vec) < embedding_dim:
            recipe_vec = np.concatenate([
                recipe_vec,
                np.zeros(embedding_dim - len(recipe_vec), dtype=np.float32)
            ])
        else:
            recipe_vec = recipe_vec[:embedding_dim]
        
        # Normalize to unit vector
        norm = np.linalg.norm(recipe_vec)
        if norm > 0:
            recipe_vec = recipe_vec / norm
        
        recipe_embeddings[recipe_id] = recipe_vec.tolist()
    
    print(f"  Computed embeddings for {len(recipe_embeddings)} recipes")
    return recipe_embeddings


def update_database_embeddings(
    supabase: Client,
    recipe_embeddings: dict[int, list[float]],
    batch_size: int = 100
) -> None:
    """Update recipe embeddings in the database."""
    print("Updating database with new embeddings...")
    
    recipe_ids = list(recipe_embeddings.keys())
    total = len(recipe_ids)
    updated = 0
    
    for i in range(0, total, batch_size):
        batch_ids = recipe_ids[i:i + batch_size]
        
        for recipe_id in batch_ids:
            embedding = recipe_embeddings[recipe_id]
            
            try:
                supabase.table("recipes").update({
                    "embedding": embedding
                }).eq("id", recipe_id).execute()
                updated += 1
            except Exception as e:
                print(f"  Error updating recipe {recipe_id}: {e}")
        
        print(f"  Updated {updated}/{total} recipes...")
    
    print(f"Successfully updated {updated} recipes")


def parse_embedding(embedding) -> np.ndarray | None:
    """
    Parse embedding from various formats (string, list, or array).
    Supabase pgvector returns embeddings as strings like '[0.1,0.2,...]'.
    """
    if embedding is None:
        return None
    
    if isinstance(embedding, np.ndarray):
        return embedding
    
    if isinstance(embedding, list):
        return np.array(embedding, dtype=np.float32)
    
    if isinstance(embedding, str):
        try:
            # pgvector format: '[0.1,0.2,0.3,...]'
            embedding = embedding.strip()
            if embedding.startswith('[') and embedding.endswith(']'):
                # Parse the string as a list of floats
                values = [float(x.strip()) for x in embedding[1:-1].split(',')]
                return np.array(values, dtype=np.float32)
        except (ValueError, AttributeError) as e:
            print(f"  Warning: Could not parse embedding: {e}")
            return None
    
    return None


def verify_embeddings(supabase: Client, sample_size: int = 5) -> None:
    """Verify embeddings by finding similar recipes."""
    print("\nVerifying embeddings with similarity search...")
    
    # Get a few random recipes
    response = supabase.table("recipes").select(
        "id, name, embedding"
    ).limit(sample_size).execute()
    
    if not response.data:
        print("  No recipes found for verification")
        return
    
    # Get all recipes for comparison
    all_response = supabase.table("recipes").select(
        "id, name, embedding"
    ).limit(100).execute()
    
    all_recipes = all_response.data
    
    for recipe in response.data:
        query_vec = parse_embedding(recipe.get("embedding"))
        if query_vec is None:
            continue
        
        similarities = []
        
        for other in all_recipes:
            if other["id"] == recipe["id"]:
                continue
            
            other_vec = parse_embedding(other.get("embedding"))
            if other_vec is None:
                continue
            
            sim = np.dot(query_vec, other_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(other_vec) + 1e-10
            )
            similarities.append((other["name"], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_3 = similarities[:3]
        
        print(f"\n  '{recipe['name']}' is similar to:")
        for name, sim in top_3:
            print(f"    - '{name}' (similarity: {sim:.3f})")


def main():
    """Main function to build and update graph embeddings."""
    print("=" * 60)
    print("GRAPH-BASED RECIPE EMBEDDING BUILDER")
    print("=" * 60)
    
    # Initialize Supabase
    supabase = get_supabase_client()
    
    # Fetch all recipes
    recipes = fetch_all_recipes(supabase)
    
    if not recipes:
        print("No recipes found in database!")
        return
    
    # Build vocabulary
    ing_to_idx, idx_to_ing, act_to_idx, idx_to_act = build_vocabulary(recipes)
    
    if not ing_to_idx:
        print("No ingredients found! Check that NER field is populated.")
        return
    
    # Build co-occurrence matrices
    ing_cooccur, recipe_features = build_cooccurrence_matrix(
        recipes, ing_to_idx, act_to_idx
    )
    
    # Compute ingredient embeddings
    ing_embeddings = compute_ingredient_embeddings(
        ing_cooccur, embedding_dim=min(128, len(ing_to_idx))
    )
    
    # Compute recipe embeddings
    recipe_embeddings = compute_recipe_embeddings(
        recipes, ing_to_idx, act_to_idx, ing_embeddings, EMBEDDING_DIM
    )
    
    # Update database
    update_database_embeddings(supabase, recipe_embeddings)
    
    # Verify with sample similarity searches
    verify_embeddings(supabase)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()