"""
Build GNN-style embeddings for recipes using graph structure and semantic features.

This script creates embeddings based on:
1. Ingredient co-occurrence (SVD-based embeddings)
2. Flavor/texture category profiles
3. Graph structure information (node/edge features)

The embeddings capture recipe similarity based on ingredients, cooking methods, and flavor profiles.
"""

import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target table
TABLE_NAME = "recipes"

# Embedding configuration
EMBEDDING_DIM = 512
INGREDIENT_DIM = 256  # Dimension for ingredient component
FLAVOR_DIM = 128      # Dimension for flavor/texture component
GRAPH_DIM = 128       # Dimension for graph structure component

# Flavor and texture categories for semantic embedding
FILTER_DICT = {
    'sweet': ['sugar', 'honey', 'maple syrup', 'dates', 'molasses', 'brown sugar', 'jam', 'jelly', 'vanilla'],
    'sour': ['vinegar', 'lemon', 'lime', 'yogurt', 'buttermilk', 'sour cream', 'tamarind', 'cranberries'],
    'umami': ['soy sauce', 'fish sauce', 'miso', 'parmesan', 'mushrooms', 'tomato paste', 'worcestershire'],
    'bitter': ['cocoa', 'coffee', 'kale', 'arugula', 'grapefruit', 'dark chocolate', 'espresso'],
    'salty': ['salt', 'soy sauce', 'fish sauce', 'olives', 'capers', 'bacon', 'parmesan', 'feta cheese'],
    'fat': ['butter', 'cream', 'oil', 'cheese', 'avocado', 'nuts', 'egg yolk', 'coconut oil'],
    'spicy': ['chili', 'pepper', 'jalapeño', 'cayenne', 'wasabi', 'horseradish', 'ginger', 'hot sauce'],
    'crunchy': ['nuts', 'breadcrumbs', 'seeds', 'crackers', 'celery', 'chips', 'croutons'],
    'creamy': ['cream', 'cream cheese', 'mascarpone', 'yogurt', 'avocado', 'mayonnaise', 'custard'],
    'starchy': ['potato', 'rice', 'pasta', 'bread', 'flour', 'corn', 'beans', 'lentils']
}

CATEGORIES = list(FILTER_DICT.keys())


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def build_category_lookup() -> dict[str, list[str]]:
    """Build reverse lookup: ingredient -> list of categories."""
    ingredient_to_categories = defaultdict(list)
    
    for category, items in FILTER_DICT.items():
        for item in items:
            item_lower = item.lower().strip()
            if category not in ingredient_to_categories[item_lower]:
                ingredient_to_categories[item_lower].append(category)
    
    return dict(ingredient_to_categories)


def classify_ingredient(ingredient: str, lookup: dict[str, list[str]]) -> list[str]:
    """Classify an ingredient into its flavor/texture categories."""
    ingredient_lower = ingredient.lower().strip()
    
    # Direct match
    if ingredient_lower in lookup:
        return lookup[ingredient_lower]
    
    # Partial match
    categories = []
    for item, cats in lookup.items():
        if item in ingredient_lower or ingredient_lower in item:
            for cat in cats:
                if cat not in categories:
                    categories.append(cat)
    
    return categories


def fetch_all_recipes(supabase: Client) -> list[dict]:
    """Fetch all recipes from the database."""
    print("📂 Fetching recipes from database...")
    
    all_recipes = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table(TABLE_NAME).select(
            "id, name, ingredients, ner, tokens, directions, graph_representation"
        ).range(offset, offset + page_size - 1).execute()
        
        if not response.data:
            break
            
        all_recipes.extend(response.data)
        offset += page_size
        print(f"   Fetched {len(all_recipes)} recipes...")
        
        if len(response.data) < page_size:
            break
    
    print(f"✅ Total recipes: {len(all_recipes)}")
    return all_recipes


def build_ingredient_vocabulary(recipes: list[dict]) -> tuple[dict[str, int], list[str]]:
    """Build ingredient vocabulary with frequency filtering."""
    print("📊 Building ingredient vocabulary...")
    
    ingredient_counts = defaultdict(int)
    
    for recipe in recipes:
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        for ing in ner:
            if ing:
                ingredient_counts[ing.lower().strip()] += 1
    
    # Filter to ingredients appearing at least twice
    min_count = 2
    ingredients = sorted([ing for ing, count in ingredient_counts.items() if count >= min_count])
    ing_to_idx = {ing: idx for idx, ing in enumerate(ingredients)}
    
    print(f"   Vocabulary size: {len(ingredients)} ingredients")
    return ing_to_idx, ingredients


def build_ingredient_embeddings(
    recipes: list[dict],
    ing_to_idx: dict[str, int],
    embedding_dim: int = 128
) -> np.ndarray:
    """Build ingredient embeddings using co-occurrence matrix and SVD."""
    print("🔧 Building ingredient co-occurrence matrix...")
    
    n_ingredients = len(ing_to_idx)
    if n_ingredients == 0:
        return np.array([])
    
    # Build co-occurrence matrix
    cooccur = np.zeros((n_ingredients, n_ingredients), dtype=np.float32)
    
    for recipe in recipes:
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
                    ing_indices.append(ing_to_idx[ing_lower])
        
        # Update co-occurrence
        for i in range(len(ing_indices)):
            for j in range(i + 1, len(ing_indices)):
                cooccur[ing_indices[i], ing_indices[j]] += 1
                cooccur[ing_indices[j], ing_indices[i]] += 1
    
    # Apply PPMI (Positive Pointwise Mutual Information)
    print("   Computing PPMI and SVD...")
    row_sums = cooccur.sum(axis=1, keepdims=True) + 1e-10
    col_sums = cooccur.sum(axis=0, keepdims=True) + 1e-10
    total = cooccur.sum() + 1e-10
    
    pmi = np.log((cooccur * total) / (row_sums * col_sums) + 1e-10)
    ppmi = np.maximum(pmi, 0)
    
    # SVD decomposition
    k = min(embedding_dim, n_ingredients - 1, ppmi.shape[1] - 1)
    if k < 1:
        embeddings = np.random.randn(n_ingredients, embedding_dim).astype(np.float32) * 0.1
    else:
        sparse_ppmi = csr_matrix(ppmi)
        U, S, Vt = svds(sparse_ppmi, k=k)
        embeddings = U * np.sqrt(S)
        
        # Pad to target dimension
        if embeddings.shape[1] < embedding_dim:
            padding = np.zeros((n_ingredients, embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
    
    print(f"   Ingredient embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def compute_flavor_profile(ner: list, lookup: dict[str, list[str]]) -> np.ndarray:
    """Compute flavor/texture profile vector for a recipe."""
    profile = np.zeros(len(CATEGORIES), dtype=np.float32)
    
    for ingredient in ner:
        categories = classify_ingredient(ingredient, lookup)
        for cat in categories:
            if cat in CATEGORIES:
                idx = CATEGORIES.index(cat)
                profile[idx] += 1.0
    
    # Normalize
    if ner:
        profile /= len(ner)
    
    return profile


def compute_graph_features(graph: dict) -> np.ndarray:
    """Extract structural features from the recipe graph."""
    features = np.zeros(16, dtype=np.float32)
    
    if not graph or not isinstance(graph, dict):
        return features
    
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Basic counts (normalized)
    features[0] = len(nodes) / 20.0  # Normalized node count
    features[1] = len(edges) / 30.0  # Normalized edge count
    
    # Node type distribution
    ingredient_nodes = sum(1 for n in nodes if n.get("type") == "ingredient")
    intermediate_nodes = sum(1 for n in nodes if n.get("type") == "intermediate")
    final_nodes = sum(1 for n in nodes if n.get("type") == "final")
    
    total_nodes = len(nodes) or 1
    features[2] = ingredient_nodes / total_nodes
    features[3] = intermediate_nodes / total_nodes
    features[4] = final_nodes / total_nodes
    
    # Edge density
    max_edges = total_nodes * (total_nodes - 1) / 2 if total_nodes > 1 else 1
    features[5] = len(edges) / max_edges if max_edges > 0 else 0
    
    # Compute node degrees
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for edge in edges:
        out_degree[edge.get("source", 0)] += 1
        in_degree[edge.get("target", 0)] += 1
    
    # Average degrees
    if nodes:
        features[6] = np.mean(list(in_degree.values())) / 5.0 if in_degree else 0
        features[7] = np.mean(list(out_degree.values())) / 5.0 if out_degree else 0
        features[8] = max(in_degree.values()) / 10.0 if in_degree else 0
        features[9] = max(out_degree.values()) / 10.0 if out_degree else 0
    
    # Action diversity (unique actions)
    actions = set(edge.get("action", "") for edge in edges)
    features[10] = len(actions) / 10.0
    
    # Step count
    max_step = max((edge.get("step", 0) for edge in edges), default=0)
    features[11] = max_step / 10.0
    
    # Metadata features
    metadata = graph.get("metadata", {})
    features[12] = 1.0 if metadata.get("fallback") else 0.0
    features[13] = metadata.get("step_count", 0) / 10.0
    
    return features


def compute_recipe_embedding(
    recipe: dict,
    ing_to_idx: dict[str, int],
    ing_embeddings: np.ndarray,
    category_lookup: dict[str, list[str]]
) -> np.ndarray:
    """
    Compute final embedding for a recipe combining:
    1. Ingredient embeddings (averaged)
    2. Flavor/texture profile
    3. Graph structure features
    """
    # Parse NER
    ner = recipe.get("ner") or []
    if isinstance(ner, str):
        try:
            ner = json.loads(ner)
        except:
            ner = []
    
    # Parse graph
    graph = recipe.get("graph_representation") or {}
    if isinstance(graph, str):
        try:
            graph = json.loads(graph)
        except:
            graph = {}
    
    # === Component 1: Ingredient embedding (average pooling) ===
    ing_embed_dim = ing_embeddings.shape[1] if len(ing_embeddings) > 0 else INGREDIENT_DIM
    ing_vec = np.zeros(ing_embed_dim, dtype=np.float32)
    ing_count = 0
    
    for ing in ner:
        if ing:
            ing_lower = ing.lower().strip()
            if ing_lower in ing_to_idx:
                idx = ing_to_idx[ing_lower]
                if idx < len(ing_embeddings):
                    ing_vec += ing_embeddings[idx]
                    ing_count += 1
    
    if ing_count > 0:
        ing_vec /= ing_count
    
    # === Component 2: Flavor profile embedding ===
    flavor_profile = compute_flavor_profile(ner, category_lookup)
    
    # Expand flavor profile to target dimension
    flavor_vec = np.zeros(FLAVOR_DIM, dtype=np.float32)
    base_dim = len(flavor_profile)
    repeat = FLAVOR_DIM // base_dim
    for i, val in enumerate(flavor_profile):
        for j in range(repeat):
            if i * repeat + j < FLAVOR_DIM:
                flavor_vec[i * repeat + j] = val
    
    # === Component 3: Graph structure embedding ===
    graph_features = compute_graph_features(graph)
    
    # Expand graph features to target dimension
    graph_vec = np.zeros(GRAPH_DIM, dtype=np.float32)
    base_dim = len(graph_features)
    repeat = GRAPH_DIM // base_dim
    for i, val in enumerate(graph_features):
        for j in range(repeat):
            if i * repeat + j < GRAPH_DIM:
                graph_vec[i * repeat + j] = val
    
    # === Combine components ===
    def resize_vector(v: np.ndarray, target_dim: int) -> np.ndarray:
        if len(v) == 0:
            return np.zeros(target_dim, dtype=np.float32)
        if len(v) >= target_dim:
            return v[:target_dim]
        else:
            return np.concatenate([v, np.zeros(target_dim - len(v), dtype=np.float32)])
    
    # Resize components to their target dimensions
    ing_component = resize_vector(ing_vec, INGREDIENT_DIM)
    flavor_component = resize_vector(flavor_vec, FLAVOR_DIM)
    graph_component = resize_vector(graph_vec, GRAPH_DIM)
    
    # Concatenate all components
    full_embedding = np.concatenate([ing_component, flavor_component, graph_component])
    
    # Ensure correct final dimension
    if len(full_embedding) < EMBEDDING_DIM:
        full_embedding = np.concatenate([
            full_embedding, 
            np.zeros(EMBEDDING_DIM - len(full_embedding), dtype=np.float32)
        ])
    elif len(full_embedding) > EMBEDDING_DIM:
        full_embedding = full_embedding[:EMBEDDING_DIM]
    
    # L2 normalize
    norm = np.linalg.norm(full_embedding)
    if norm > 0:
        full_embedding /= norm
    
    return full_embedding


def compute_all_embeddings(
    recipes: list[dict],
    ing_to_idx: dict[str, int],
    ing_embeddings: np.ndarray,
    category_lookup: dict[str, list[str]]
) -> dict[int, list[float]]:
    """Compute embeddings for all recipes."""
    print("🧠 Computing recipe embeddings...")
    
    embeddings = {}
    
    for idx, recipe in enumerate(recipes):
        if (idx + 1) % 500 == 0:
            print(f"   Processing recipe {idx + 1}/{len(recipes)}...")
        
        embedding = compute_recipe_embedding(
            recipe, ing_to_idx, ing_embeddings, category_lookup
        )
        embeddings[recipe["id"]] = embedding.tolist()
    
    print(f"✅ Computed embeddings for {len(embeddings)} recipes")
    return embeddings


def update_database_embeddings(
    supabase: Client,
    embeddings: dict[int, list[float]],
    batch_size: int = 100
) -> int:
    """Update recipe embeddings in the database."""
    print("📤 Updating embeddings in database...")
    
    recipe_ids = list(embeddings.keys())
    total = len(recipe_ids)
    updated = 0
    errors = 0
    
    for i in range(0, total, batch_size):
        batch_ids = recipe_ids[i:i + batch_size]
        
        for recipe_id in batch_ids:
            embedding = embeddings[recipe_id]
            
            try:
                supabase.table(TABLE_NAME).update({
                    "embedding": embedding
                }).eq("id", recipe_id).execute()
                updated += 1
            except Exception as e:
                print(f"   ❌ Error updating recipe {recipe_id}: {e}")
                errors += 1
        
        print(f"   Updated {updated}/{total} recipes...")
    
    print(f"✅ Successfully updated {updated} recipes ({errors} errors)")
    return updated


def build_gnn_embeddings(limit: int = None):
    """Main function to build and store GNN-style embeddings."""
    print("=" * 60)
    print("GNN-STYLE RECIPE EMBEDDING BUILDER")
    print("=" * 60)
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"  - Ingredient component: {INGREDIENT_DIM}")
    print(f"  - Flavor component: {FLAVOR_DIM}")
    print(f"  - Graph component: {GRAPH_DIM}")
    print("=" * 60)
    
    # Initialize
    supabase = get_supabase_client()
    category_lookup = build_category_lookup()
    
    # Fetch recipes
    recipes = fetch_all_recipes(supabase)
    
    if limit:
        recipes = recipes[:limit]
        print(f"   Limiting to {limit} recipes")
    
    if not recipes:
        print("❌ No recipes found in database")
        return
    
    # Build ingredient vocabulary and embeddings
    ing_to_idx, ingredients = build_ingredient_vocabulary(recipes)
    
    if not ing_to_idx:
        print("❌ No ingredients found")
        return
    
    # Build ingredient embeddings
    ing_embeddings = build_ingredient_embeddings(
        recipes, ing_to_idx, embedding_dim=min(128, len(ing_to_idx) - 1)
    )
    
    # Compute all recipe embeddings
    embeddings = compute_all_embeddings(
        recipes, ing_to_idx, ing_embeddings, category_lookup
    )
    
    # Update database
    update_database_embeddings(supabase, embeddings)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\n📌 Next step: Run 5_validate_embeddings.py")


def analyze_embeddings(sample_size: int = 5):
    """Analyze embedding distribution and quality."""
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)
    
    supabase = get_supabase_client()
    
    response = supabase.table(TABLE_NAME).select(
        "id, name, embedding"
    ).limit(100).execute()
    
    embeddings = []
    for recipe in response.data:
        emb = recipe.get("embedding")
        if emb:
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb.replace('[', '').replace(']', '').split(','))
                    emb = [float(x) for x in emb]
                except:
                    continue
            embeddings.append(np.array(emb))
    
    if not embeddings:
        print("❌ No embeddings found")
        return
    
    embeddings = np.array(embeddings)
    
    print(f"\nEmbedding statistics (n={len(embeddings)}):")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # Check norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Min: {norms.min():.4f}")
    print(f"  Max: {norms.max():.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--analyze":
            analyze_embeddings()
        elif sys.argv[1] == "--limit":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            build_gnn_embeddings(limit=limit)
        else:
            print("Usage:")
            print("  python 4_build_gnn_embeddings.py           # Build all embeddings")
            print("  python 4_build_gnn_embeddings.py --limit N # Build for first N recipes")
            print("  python 4_build_gnn_embeddings.py --analyze # Analyze existing embeddings")
    else:
        build_gnn_embeddings()
