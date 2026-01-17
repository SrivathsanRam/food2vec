"""
Validate GNN embeddings for recipe similarity.

This script validates that the embeddings capture meaningful recipe similarity by:
1. Computing cosine similarity between recipes
2. Checking that similar recipes (by ingredients) have high similarity scores
3. Performing clustering analysis
4. Testing nearest neighbor retrieval
"""

import os
import json
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Target table
TABLE_NAME = "recipes"


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_embedding(embedding) -> np.ndarray | None:
    """Parse embedding from various formats."""
    if embedding is None:
        return None
    
    if isinstance(embedding, np.ndarray):
        return embedding
    
    if isinstance(embedding, list):
        return np.array(embedding, dtype=np.float32)
    
    if isinstance(embedding, str):
        try:
            embedding = embedding.strip()
            if embedding.startswith('[') and embedding.endswith(']'):
                values = [float(x.strip()) for x in embedding[1:-1].split(',')]
                return np.array(values, dtype=np.float32)
        except (ValueError, AttributeError):
            return None
    
    return None


def fetch_recipes_with_embeddings(supabase: Client, limit: int = 500) -> list[dict]:
    """Fetch recipes with their embeddings."""
    print(f"📂 Fetching recipes with embeddings (limit: {limit})...")
    
    response = supabase.table(TABLE_NAME).select(
        "id, name, ner, embedding, graph_representation"
    ).limit(limit).execute()
    
    recipes = []
    for recipe in response.data:
        emb = parse_embedding(recipe.get("embedding"))
        if emb is not None and np.any(emb != 0):  # Skip zero embeddings
            recipe["embedding_array"] = emb
            recipes.append(recipe)
    
    print(f"✅ Found {len(recipes)} recipes with valid embeddings")
    return recipes


def compute_ingredient_overlap(ner1: list, ner2: list) -> float:
    """Compute Jaccard similarity between ingredient lists."""
    if not ner1 or not ner2:
        return 0.0
    
    set1 = set(ing.lower().strip() for ing in ner1)
    set2 = set(ing.lower().strip() for ing in ner2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def validate_similarity_correlation(recipes: list[dict]) -> dict:
    """
    Validate that embedding similarity correlates with ingredient overlap.
    
    Good embeddings should show:
    - High embedding similarity for recipes with similar ingredients
    - Low embedding similarity for recipes with different ingredients
    """
    print("\n" + "=" * 60)
    print("SIMILARITY CORRELATION VALIDATION")
    print("=" * 60)
    
    n = min(len(recipes), 100)  # Limit for computational efficiency
    
    embedding_sims = []
    ingredient_sims = []
    
    for i in range(n):
        ner_i = recipes[i].get("ner", [])
        if isinstance(ner_i, str):
            try:
                ner_i = json.loads(ner_i)
            except:
                ner_i = []
        
        emb_i = recipes[i]["embedding_array"]
        
        for j in range(i + 1, n):
            ner_j = recipes[j].get("ner", [])
            if isinstance(ner_j, str):
                try:
                    ner_j = json.loads(ner_j)
                except:
                    ner_j = []
            
            emb_j = recipes[j]["embedding_array"]
            
            # Compute similarities
            emb_sim = cosine_similarity([emb_i], [emb_j])[0, 0]
            ing_sim = compute_ingredient_overlap(ner_i, ner_j)
            
            embedding_sims.append(emb_sim)
            ingredient_sims.append(ing_sim)
    
    # Compute correlation
    embedding_sims = np.array(embedding_sims)
    ingredient_sims = np.array(ingredient_sims)
    
    if len(embedding_sims) > 1:
        correlation = np.corrcoef(embedding_sims, ingredient_sims)[0, 1]
    else:
        correlation = 0.0
    
    # Compute statistics by ingredient similarity buckets
    buckets = {
        "high_ing_sim (>0.3)": [],
        "med_ing_sim (0.1-0.3)": [],
        "low_ing_sim (<0.1)": []
    }
    
    for emb_sim, ing_sim in zip(embedding_sims, ingredient_sims):
        if ing_sim > 0.3:
            buckets["high_ing_sim (>0.3)"].append(emb_sim)
        elif ing_sim > 0.1:
            buckets["med_ing_sim (0.1-0.3)"].append(emb_sim)
        else:
            buckets["low_ing_sim (<0.1)"].append(emb_sim)
    
    print(f"\nCorrelation between embedding similarity and ingredient overlap:")
    print(f"  Pearson correlation: {correlation:.4f}")
    
    print(f"\nEmbedding similarity by ingredient overlap bucket:")
    for bucket, sims in buckets.items():
        if sims:
            print(f"  {bucket}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, n={len(sims)}")
    
    # Quality assessment
    quality = "GOOD" if correlation > 0.3 else "MODERATE" if correlation > 0.1 else "POOR"
    print(f"\n📊 Embedding quality assessment: {quality}")
    
    return {
        "correlation": correlation,
        "buckets": {k: {"mean": np.mean(v) if v else 0, "count": len(v)} for k, v in buckets.items()},
        "quality": quality
    }


def validate_nearest_neighbors(recipes: list[dict], k: int = 5) -> dict:
    """
    Validate nearest neighbor retrieval.
    
    For each recipe, find its k nearest neighbors and check:
    - How many neighbors share common ingredients
    - Average ingredient overlap with neighbors
    """
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBOR VALIDATION")
    print("=" * 60)
    
    n = min(len(recipes), 50)  # Sample size
    
    # Build embedding matrix
    embeddings = np.array([r["embedding_array"] for r in recipes[:n]])
    
    # Compute all pairwise similarities
    sim_matrix = cosine_similarity(embeddings)
    
    overlap_scores = []
    neighbor_stats = []
    
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # Exclude self
        neighbor_indices = np.argsort(sims)[-k:][::-1]
        
        ner_i = recipes[i].get("ner", [])
        if isinstance(ner_i, str):
            try:
                ner_i = json.loads(ner_i)
            except:
                ner_i = []
        
        overlaps = []
        for j in neighbor_indices:
            ner_j = recipes[j].get("ner", [])
            if isinstance(ner_j, str):
                try:
                    ner_j = json.loads(ner_j)
                except:
                    ner_j = []
            
            overlap = compute_ingredient_overlap(ner_i, ner_j)
            overlaps.append(overlap)
        
        overlap_scores.extend(overlaps)
        neighbor_stats.append({
            "recipe": recipes[i]["name"][:40],
            "mean_overlap": np.mean(overlaps),
            "max_overlap": max(overlaps) if overlaps else 0
        })
    
    avg_overlap = np.mean(overlap_scores)
    
    print(f"\nNearest neighbor ingredient overlap (k={k}):")
    print(f"  Average overlap: {avg_overlap:.4f}")
    print(f"  Max overlap: {max(overlap_scores):.4f}")
    print(f"  Min overlap: {min(overlap_scores):.4f}")
    
    # Show some examples
    print(f"\nSample nearest neighbor results:")
    for stat in sorted(neighbor_stats, key=lambda x: x["mean_overlap"], reverse=True)[:3]:
        print(f"  {stat['recipe']}: avg_overlap={stat['mean_overlap']:.3f}")
    
    return {
        "avg_overlap": avg_overlap,
        "sample_stats": neighbor_stats[:5]
    }


def validate_clustering(recipes: list[dict], n_clusters: int = 10) -> dict:
    """
    Validate embeddings using clustering analysis.
    
    Good embeddings should form clusters where recipes share similar characteristics.
    """
    print("\n" + "=" * 60)
    print("CLUSTERING VALIDATION")
    print("=" * 60)
    
    n = min(len(recipes), 200)
    embeddings = np.array([r["embedding_array"] for r in recipes[:n]])
    
    # Perform K-means clustering
    print(f"\nPerforming K-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    cluster_analysis = defaultdict(list)
    
    for i, cluster_id in enumerate(clusters):
        ner = recipes[i].get("ner", [])
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        cluster_analysis[cluster_id].append({
            "name": recipes[i]["name"],
            "ingredients": ner
        })
    
    # Compute intra-cluster similarity
    print(f"\nCluster analysis:")
    cluster_cohesions = []
    
    for cluster_id in range(n_clusters):
        members = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        if len(members) < 2:
            continue
        
        # Compute average pairwise similarity within cluster
        cluster_embeddings = embeddings[members]
        sim_matrix = cosine_similarity(cluster_embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices(len(members), k=1)]
        avg_sim = np.mean(upper_tri) if len(upper_tri) > 0 else 0
        cluster_cohesions.append(avg_sim)
        
        # Get common ingredients in cluster
        all_ingredients = []
        for m in members:
            ner = recipes[m].get("ner", [])
            if isinstance(ner, str):
                try:
                    ner = json.loads(ner)
                except:
                    ner = []
            all_ingredients.extend([ing.lower() for ing in ner])
        
        from collections import Counter
        common = Counter(all_ingredients).most_common(3)
        common_str = ", ".join([f"{ing}({cnt})" for ing, cnt in common])
        
        print(f"  Cluster {cluster_id}: {len(members)} recipes, cohesion={avg_sim:.3f}")
        print(f"    Common ingredients: {common_str}")
    
    avg_cohesion = np.mean(cluster_cohesions) if cluster_cohesions else 0
    print(f"\nAverage cluster cohesion: {avg_cohesion:.4f}")
    
    return {
        "n_clusters": n_clusters,
        "avg_cohesion": avg_cohesion,
        "cluster_sizes": [len([c for c in clusters if c == i]) for i in range(n_clusters)]
    }


def validate_embedding_space(recipes: list[dict]) -> dict:
    """
    Validate embedding space properties.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING SPACE VALIDATION")
    print("=" * 60)
    
    embeddings = np.array([r["embedding_array"] for r in recipes])
    
    # Basic statistics
    print(f"\nEmbedding dimensions: {embeddings.shape[1]}")
    print(f"Number of recipes: {embeddings.shape[0]}")
    
    # Check for degenerate embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    zero_embeddings = np.sum(norms < 0.01)
    
    print(f"\nNorm statistics:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std: {norms.std():.4f}")
    print(f"  Min: {norms.min():.4f}")
    print(f"  Max: {norms.max():.4f}")
    print(f"  Near-zero embeddings: {zero_embeddings}")
    
    # Check dimensionality coverage
    # Using PCA to see how many dimensions are actually used
    pca = PCA()
    pca.fit(embeddings)
    
    # Find dimensions that explain 90% of variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_dims_90 = np.argmax(cumsum >= 0.90) + 1
    n_dims_95 = np.argmax(cumsum >= 0.95) + 1
    
    print(f"\nDimensionality analysis (PCA):")
    print(f"  Dimensions for 90% variance: {n_dims_90}")
    print(f"  Dimensions for 95% variance: {n_dims_95}")
    print(f"  Effective dimensionality ratio: {n_dims_95 / embeddings.shape[1]:.2%}")
    
    # Check for uniform distribution
    all_sims = cosine_similarity(embeddings[:100])
    upper_tri = all_sims[np.triu_indices(min(100, len(recipes)), k=1)]
    
    print(f"\nSimilarity distribution:")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Std: {np.std(upper_tri):.4f}")
    print(f"  Min: {np.min(upper_tri):.4f}")
    print(f"  Max: {np.max(upper_tri):.4f}")
    
    return {
        "n_recipes": len(recipes),
        "embedding_dim": embeddings.shape[1],
        "zero_embeddings": zero_embeddings,
        "effective_dims_95": n_dims_95,
        "mean_similarity": float(np.mean(upper_tri)),
        "similarity_std": float(np.std(upper_tri))
    }


def find_similar_recipes(supabase: Client, recipe_name: str, k: int = 5):
    """Find similar recipes to a given recipe name."""
    print("\n" + "=" * 60)
    print(f"SIMILAR RECIPES TO: {recipe_name}")
    print("=" * 60)
    
    # Find the target recipe
    response = supabase.table(TABLE_NAME).select(
        "id, name, ner, embedding"
    ).ilike("name", f"%{recipe_name}%").limit(1).execute()
    
    if not response.data:
        print(f"❌ Recipe not found: {recipe_name}")
        return
    
    target = response.data[0]
    target_emb = parse_embedding(target.get("embedding"))
    
    if target_emb is None:
        print("❌ Target recipe has no embedding")
        return
    
    target_ner = target.get("ner", [])
    if isinstance(target_ner, str):
        try:
            target_ner = json.loads(target_ner)
        except:
            target_ner = []
    
    print(f"\nTarget: {target['name']}")
    print(f"Ingredients: {', '.join(target_ner[:5])}...")
    
    # Get all recipes
    all_response = supabase.table(TABLE_NAME).select(
        "id, name, ner, embedding"
    ).limit(500).execute()
    
    # Compute similarities
    similarities = []
    for recipe in all_response.data:
        if recipe["id"] == target["id"]:
            continue
        
        emb = parse_embedding(recipe.get("embedding"))
        if emb is None:
            continue
        
        sim = cosine_similarity([target_emb], [emb])[0, 0]
        
        ner = recipe.get("ner", [])
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        ing_overlap = compute_ingredient_overlap(target_ner, ner)
        
        similarities.append({
            "name": recipe["name"],
            "similarity": sim,
            "ingredient_overlap": ing_overlap,
            "ingredients": ner
        })
    
    # Sort by embedding similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"\nTop {k} similar recipes (by embedding):")
    for i, sim in enumerate(similarities[:k]):
        print(f"\n  {i+1}. {sim['name'][:50]}")
        print(f"     Embedding similarity: {sim['similarity']:.4f}")
        print(f"     Ingredient overlap: {sim['ingredient_overlap']:.4f}")
        shared = set(target_ner) & set(sim['ingredients'])
        if shared:
            print(f"     Shared ingredients: {', '.join(list(shared)[:5])}")


def run_full_validation():
    """Run complete validation suite."""
    print("=" * 60)
    print("GNN EMBEDDING VALIDATION SUITE")
    print("=" * 60)
    
    supabase = get_supabase_client()
    recipes = fetch_recipes_with_embeddings(supabase, limit=500)
    
    if len(recipes) < 10:
        print("❌ Not enough recipes with valid embeddings for validation")
        print("   Run 4_build_gnn_embeddings.py first to generate embeddings")
        return
    
    results = {}
    
    # Run all validations
    results["space"] = validate_embedding_space(recipes)
    results["similarity"] = validate_similarity_correlation(recipes)
    results["neighbors"] = validate_nearest_neighbors(recipes)
    results["clustering"] = validate_clustering(recipes)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Embedding Space:")
    print(f"   - Recipes with embeddings: {results['space']['n_recipes']}")
    print(f"   - Embedding dimension: {results['space']['embedding_dim']}")
    print(f"   - Effective dimensions (95%): {results['space']['effective_dims_95']}")
    
    print(f"\n📈 Similarity Correlation:")
    print(f"   - Correlation with ingredients: {results['similarity']['correlation']:.4f}")
    print(f"   - Quality: {results['similarity']['quality']}")
    
    print(f"\n🔍 Nearest Neighbors:")
    print(f"   - Avg ingredient overlap with neighbors: {results['neighbors']['avg_overlap']:.4f}")
    
    print(f"\n📦 Clustering:")
    print(f"   - Avg cluster cohesion: {results['clustering']['avg_cohesion']:.4f}")
    
    # Overall assessment
    overall_quality = "GOOD"
    issues = []
    
    if results['similarity']['correlation'] < 0.1:
        overall_quality = "POOR"
        issues.append("Low correlation with ingredient overlap")
    elif results['similarity']['correlation'] < 0.3:
        overall_quality = "MODERATE"
        issues.append("Moderate correlation with ingredient overlap")
    
    if results['clustering']['avg_cohesion'] < 0.5:
        issues.append("Low cluster cohesion")
        if overall_quality == "GOOD":
            overall_quality = "MODERATE"
    
    print(f"\n🎯 OVERALL QUALITY: {overall_quality}")
    if issues:
        print("   Issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--similar":
            if len(sys.argv) > 2:
                recipe_name = " ".join(sys.argv[2:])
                supabase = get_supabase_client()
                find_similar_recipes(supabase, recipe_name)
            else:
                print("Usage: python 5_validate_embeddings.py --similar <recipe name>")
        elif sys.argv[1] == "--quick":
            supabase = get_supabase_client()
            recipes = fetch_recipes_with_embeddings(supabase, limit=100)
            validate_embedding_space(recipes)
        else:
            print("Usage:")
            print("  python 5_validate_embeddings.py              # Run full validation")
            print("  python 5_validate_embeddings.py --quick      # Quick validation")
            print("  python 5_validate_embeddings.py --similar X  # Find similar to recipe X")
    else:
        run_full_validation()
