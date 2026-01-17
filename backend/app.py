from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Vector dimension for embeddings
EMBEDDING_DIM = 512


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def stable_hash(s: str) -> int:
    """Generate a stable hash for a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def generate_embedding(tokens: list[str]) -> list[float]:
    """Generate a deterministic mock embedding from tokens."""
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for t in tokens:
        h = stable_hash(t)
        idx = h % EMBEDDING_DIM
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        if t.startswith("ACT_"):
            weight = 2.0
        elif t.startswith("ING_"):
            weight = 1.5
        else:
            weight = 1.0
        v[idx] += sign * weight
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v.tolist()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    supabase = get_supabase_client()
    db_status = "connected" if supabase else "not configured"
    return jsonify({
        "status": "healthy", 
        "message": "Food2Vec API is running!",
        "database": db_status
    })


@app.route('/api/food-names', methods=['GET'])
def get_all_food_names():
    """Return all food names with a version hash for cache invalidation."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"names": [], "version": "", "error": "Database not configured"})
        
        # Get all recipe names
        result = supabase.table("recipes") \
            .select("name") \
            .order("name") \
            .execute()
        
        names = [row["name"] for row in result.data]
        
        # Generate a version hash based on the names list
        # This will change when items are added/removed/modified
        version = hashlib.md5("".join(names).encode()).hexdigest()[:12]
        
        return jsonify({
            "names": names,
            "version": version,
            "count": len(names)
        })
        
    except Exception as e:
        print(f"Get food names error: {e}")
        return jsonify({"names": [], "version": "", "error": str(e)})


@app.route('/api/food-names/version', methods=['GET'])
def get_food_names_version():
    """Return just the version hash to check if cache needs refresh."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"version": "", "count": 0})
        
        # Get count and a sample to generate version
        result = supabase.table("recipes") \
            .select("name") \
            .order("name") \
            .execute()
        
        names = [row["name"] for row in result.data]
        version = hashlib.md5("".join(names).encode()).hexdigest()[:12]
        
        return jsonify({
            "version": version,
            "count": len(names)
        })
        
    except Exception as e:
        print(f"Get version error: {e}")
        return jsonify({"version": "", "count": 0})


@app.route('/api/autocomplete', methods=['GET'])
def autocomplete():
    """Return recipe title suggestions based on query from Supabase."""
    query = request.args.get('q', '').strip()
    
    if not query or len(query) < 1:
        return jsonify({"suggestions": []})
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"suggestions": [], "error": "Database not configured"})
        
        # Search for recipes with titles matching the query (case-insensitive)
        result = supabase.table("recipes") \
            .select("name") \
            .ilike("name", f"%{query}%") \
            .limit(10) \
            .execute()
        
        suggestions = [row["name"] for row in result.data]
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        print(f"Autocomplete error: {e}")
        return jsonify({"suggestions": [], "error": str(e)})


@app.route('/api/search', methods=['POST'])
def search_food():
    """Search for recipes by title using Supabase."""
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # Search for recipes with titles matching the query
        result = supabase.table("recipes") \
            .select("id, name, ingredients, directions, ner, graph_representation") \
            .ilike("name", f"%{query}%") \
            .limit(top_k) \
            .execute()
        
        # Format results
        results = []
        for row in result.data:
            # Calculate a simple relevance score based on match position
            name_lower = row["name"].lower()
            query_lower = query.lower()
            if name_lower.startswith(query_lower):
                score = 0.95
            elif query_lower in name_lower:
                score = 0.80
            else:
                score = 0.60
            
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row["ingredients"],
                "directions": row["directions"],
                "ner": row.get("ner", []),
                "graph": row.get("graph_representation", {}),
                "score": score,
                "category": categorize_recipe(row.get("ner", []))
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "message": "Search completed successfully"
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/vector', methods=['POST'])
def search_by_vector():
    """Search for similar recipes using vector similarity."""
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    top_k = data.get('top_k', 10)
    
    if not ingredients:
        return jsonify({"error": "Ingredients list is required"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # Generate tokens from ingredients
        tokens = [f"ING_{ing.strip().lower().replace(' ', '_')}" for ing in ingredients]
        
        # Generate embedding
        query_embedding = generate_embedding(tokens)
        
        # Use RPC to call the similarity search function
        result = supabase.rpc(
            "search_recipes",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": top_k
            }
        ).execute()
        
        results = []
        for row in result.data:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row["ingredients"],
                "directions": row["directions"],
                "graph": row.get("graph_representation", {}),
                "score": row.get("similarity", 0),
                "category": "Recipe"
            })
        
        return jsonify({
            "query_ingredients": ingredients,
            "results": results,
            "count": len(results),
            "message": "Vector search completed successfully"
        })
        
    except Exception as e:
        print(f"Vector search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/ingredient', methods=['GET'])
def search_by_ingredient():
    """Search for recipes containing a specific ingredient."""
    ingredient = request.args.get('q', '').strip()
    limit = request.args.get('limit', 20, type=int)
    
    if not ingredient:
        return jsonify({"error": "Ingredient query is required"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # Search in the NER field (simplified ingredient names)
        result = supabase.table("recipes") \
            .select("id, name, ingredients, directions, ner") \
            .contains("ner", [ingredient.lower()]) \
            .limit(limit) \
            .execute()
        
        # If no exact match, try partial match
        if not result.data:
            result = supabase.table("recipes") \
                .select("id, name, ingredients, directions, ner") \
                .ilike("ingredients", f"%{ingredient}%") \
                .limit(limit) \
                .execute()
        
        results = []
        for row in result.data:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row["ingredients"],
                "directions": row["directions"],
                "ner": row.get("ner", []),
                "category": categorize_recipe(row.get("ner", []))
            })
        
        return jsonify({
            "query_ingredient": ingredient,
            "results": results,
            "count": len(results),
            "message": "Ingredient search completed successfully"
        })
        
    except Exception as e:
        print(f"Ingredient search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recipe/<int:recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    """Get a specific recipe by ID."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        result = supabase.table("recipes") \
            .select("*") \
            .eq("id", recipe_id) \
            .single() \
            .execute()
        
        if not result.data:
            return jsonify({"error": "Recipe not found"}), 404
        
        recipe = result.data
        return jsonify({
            "id": recipe["id"],
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
            "ner": recipe.get("ner", []),
            "graph": recipe.get("graph_representation", {}),
            "tokens": recipe.get("tokens", []),
            "link": recipe.get("link", ""),
            "source": recipe.get("source", "")
        })
        
    except Exception as e:
        print(f"Get recipe error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recipes/random', methods=['GET'])
def get_random_recipes():
    """Get random recipes for homepage display."""
    limit = request.args.get('limit', 6, type=int)
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # Get random recipes (using order by random)
        result = supabase.table("recipes") \
            .select("id, name, ingredients, ner") \
            .limit(limit) \
            .execute()
        
        results = []
        for row in result.data:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row["ingredients"][:3] if row["ingredients"] else [],  # First 3 ingredients
                "ner": row.get("ner", []),
                "category": categorize_recipe(row.get("ner", []))
            })
        
        return jsonify({
            "recipes": results,
            "count": len(results)
        })
        
    except Exception as e:
        print(f"Random recipes error: {e}")
        return jsonify({"error": str(e)}), 500


def categorize_recipe(ner_ingredients: list) -> str:
    """Categorize a recipe based on its ingredients."""
    if not ner_ingredients:
        return "Other"
    
    ner_lower = [ing.lower() for ing in ner_ingredients]
    
    # Check for dessert ingredients
    dessert_keywords = ["sugar", "chocolate", "vanilla", "flour", "butter", "cream cheese", "cocoa"]
    dessert_count = sum(1 for kw in dessert_keywords if any(kw in ing for ing in ner_lower))
    
    # Check for meat/protein
    protein_keywords = ["chicken", "beef", "pork", "fish", "salmon", "shrimp", "bacon", "sausage"]
    protein_count = sum(1 for kw in protein_keywords if any(kw in ing for ing in ner_lower))
    
    # Check for vegetables
    veggie_keywords = ["carrot", "broccoli", "pepper", "onion", "tomato", "celery", "lettuce"]
    veggie_count = sum(1 for kw in veggie_keywords if any(kw in ing for ing in ner_lower))
    
    if dessert_count >= 3:
        return "Desserts"
    elif protein_count >= 1:
        return "Main Course"
    elif veggie_count >= 2:
        return "Salads & Sides"
    else:
        return "Appetizers"


if __name__ == '__main__':
    app.run(debug=True, port=5000)
