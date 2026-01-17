from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from supabase import create_client, Client
from openapi import *

load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "recipes_duplicate_GNN2"

# Vector dimension for embeddings
EMBEDDING_DIM = 512
NODE_FEATURE_DIM = 64  # Truncated MiniLM dimension (matches training)
HIDDEN_DIM = 256

# ============== Global Model Cache (loaded once at startup) ==============
_gnn_model = None
_sentence_encoder = None
_ingredient_cache = {}
_model_loaded = False
_node_feature_dim = NODE_FEATURE_DIM  # Will be updated from config


# ============== GNN Model Definition ==============

class RecipeGAT(nn.Module):
    """Graph Attention Network for recipe embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.global_mean_pool = global_mean_pool
    
    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.global_mean_pool(x, batch)
        return x


def load_models():
    """Load GNN model and sentence encoder at startup."""
    global _gnn_model, _sentence_encoder, _model_loaded, _node_feature_dim
    
    if _model_loaded:
        return True
    
    try:
        # Load sentence encoder
        from sentence_transformers import SentenceTransformer
        _sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Loaded sentence encoder")
        
        # Load GNN model
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(model_dir, 'recipe_gnn.pt')
        config_path = os.path.join(model_dir, 'recipe_gnn_config.json')
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update node feature dim from config
            _node_feature_dim = config.get("input_dim", NODE_FEATURE_DIM)
            
            # Initialize model with saved config
            _gnn_model = RecipeGAT(
                input_dim=_node_feature_dim,
                hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
                output_dim=config.get("output_dim", EMBEDDING_DIM)
            )
            _gnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            _gnn_model.eval()
            print(f"✅ Loaded GNN model from {model_path} (input_dim={_node_feature_dim})")
        else:
            print(f"⚠️ No saved model found at {model_path}, using untrained model")
            _node_feature_dim = NODE_FEATURE_DIM
            _gnn_model = RecipeGAT(_node_feature_dim, HIDDEN_DIM, EMBEDDING_DIM)
            _gnn_model.eval()
        
        _model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


def get_ingredient_embedding(ingredient: str) -> np.ndarray:
    """Get or compute embedding for an ingredient (cached)."""
    global _ingredient_cache, _sentence_encoder, _node_feature_dim
    
    ingredient_lower = ingredient.lower().strip()
    
    if ingredient_lower not in _ingredient_cache:
        if _sentence_encoder is None:
            load_models()
        # Encode and truncate to match training dimension
        full_embedding = _sentence_encoder.encode(ingredient_lower, convert_to_numpy=True)
        truncated = full_embedding[:_node_feature_dim].astype(np.float32)
        _ingredient_cache[ingredient_lower] = truncated
    
    return _ingredient_cache[ingredient_lower]


def graph_to_pyg_data(graph: dict):
    """Convert recipe graph to PyTorch Geometric Data object."""
    from torch_geometric.data import Data
    global _node_feature_dim
    
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    if not nodes:
        return None
    
    # Build node features
    node_features = []
    for node in nodes:
        if node.get("type") == "ingredient":
            label = node.get("label", "unknown")
            feat = get_ingredient_embedding(label)
        else:
            feat = np.zeros(_node_feature_dim, dtype=np.float32)
            if node.get("type") == "intermediate":
                feat[0] = 1.0
                feat[1] = 0.5
            elif node.get("type") == "final":
                feat[0] = 0.5
                feat[1] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Build edge indices
    edge_list = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is not None and target is not None:
            edge_list.append([source, target])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


def generate_gnn_embedding(graph: dict) -> list[float]:
    """Generate GNN embedding for a recipe graph."""
    global _gnn_model
    
    if not graph or not graph.get("nodes"):
        return [0.0] * EMBEDDING_DIM
    
    try:
        if _gnn_model is None:
            load_models()
        
        data = graph_to_pyg_data(graph)
        if data is None:
            return [0.0] * EMBEDDING_DIM
        
        # Add batch dimension
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        with torch.no_grad():
            embedding = _gnn_model(data.x, data.edge_index, data.batch)
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.squeeze().tolist()
    
    except Exception as e:
        print(f"Error generating GNN embedding: {e}")
        return [0.0] * EMBEDDING_DIM


# ============== Existing Helper Functions ==============

def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def stable_hash(s: str) -> int:
    """Generate a stable hash for a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def generate_embedding(tokens: list[str]) -> list[float]:
    """Generate a deterministic mock embedding from tokens (legacy fallback)."""
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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Handle string embeddings from Supabase
    if isinstance(vec1, str):
        vec1 = json.loads(vec1)
    if isinstance(vec2, str):
        vec2 = json.loads(vec2)
    
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def categorize_recipe(ner_ingredients: list) -> str:
    """Categorize a recipe based on its ingredients."""
    if not ner_ingredients:
        return "Other"
    
    ner_lower = [ing.lower() for ing in ner_ingredients]
    
    dessert_keywords = ["sugar", "chocolate", "vanilla", "flour", "butter", "cream cheese", "cocoa"]
    dessert_count = sum(1 for kw in dessert_keywords if any(kw in ing for ing in ner_lower))
    
    protein_keywords = ["chicken", "beef", "pork", "fish", "salmon", "shrimp", "bacon", "sausage"]
    protein_count = sum(1 for kw in protein_keywords if any(kw in ing for ing in ner_lower))
    
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


# ============== API Endpoints ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    supabase = get_supabase_client()
    db_status = "connected" if supabase else "not configured"
    model_status = "loaded" if _model_loaded else "not loaded"
    return jsonify({
        "status": "healthy", 
        "message": "Food2Vec API is running!",
        "database": db_status,
        "gnn_model": model_status
    })


@app.route('/api/food-names', methods=['GET'])
def get_all_food_names():
    """Return all food names with a version hash for cache invalidation."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"names": [], "version": "", "error": "Database not configured"})
        
        # Get all recipe names
        result = supabase.table(TABLE_NAME) \
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
        result = supabase.table(TABLE_NAME) \
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
        result = supabase.table(TABLE_NAME) \
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
    """Search for recipes by name, then rank by GNN embedding cosine similarity."""
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # First, find the queried recipe to get its embedding
        query_result = supabase.table(TABLE_NAME) \
            .select("id, embedding") \
            .ilike("name", f"%{query}%") \
            .limit(1) \
            .execute()
        
        query_embedding = None
        query_recipe_id = None
        if query_result.data and query_result.data[0].get("embedding"):
            query_embedding = query_result.data[0]["embedding"]
            query_recipe_id = query_result.data[0]["id"]
        
        # If no embedding found, fall back to text search only
        if not query_embedding:
            result = supabase.table(TABLE_NAME) \
                .select("id, name, ingredients, directions, ner, graph_representation, embedding") \
                .ilike("name", f"%{query}%") \
                .limit(top_k) \
                .execute()
            
            results = []
            for row in result.data:
                results.append({
                    "id": row["id"],
                    "name": row["name"],
                    "ingredients": row["ingredients"],
                    "directions": row["directions"],
                    "ner": row.get("ner", []),
                    "graph": row.get("graph_representation", {}),
                    "score": 0.5,  # Default score when no embedding
                    "category": categorize_recipe(row.get("ner", []))
                })
            
            return jsonify({
                "query": query,
                "results": results,
                "count": len(results),
                "message": "Text search (no embedding found for query)"
            })
        
        # Fetch all recipes with embeddings to compute similarity
        # Get more than top_k to ensure we have enough after filtering
        all_recipes = supabase.table(TABLE_NAME) \
            .select("id, name, ingredients, directions, ner, graph_representation, embedding") \
            .not_.is_("embedding", "null") \
            .limit(500) \
            .execute()
        
        # Calculate cosine similarity for each recipe
        results = []
        for row in all_recipes.data:
            # Skip the query recipe itself
            if row["id"] == query_recipe_id:
                continue
            
            recipe_embedding = row.get("embedding")
            if recipe_embedding:
                # Compute cosine similarity (already normalized embeddings)
                score = cosine_similarity(query_embedding, recipe_embedding)
                # Keep raw cosine similarity (0 to 1 for normalized vectors)
                # Don't normalize again - our embeddings are already L2 normalized
            else:
                score = 0.0
            
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row["ingredients"],
                "directions": row["directions"],
                "ner": row.get("ner", []),
                "graph": row.get("graph_representation", {}),
                "score": round(score, 4),
                "category": categorize_recipe(row.get("ner", []))
            })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "message": "GNN vector similarity search completed"
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
        result = supabase.table(TABLE_NAME) \
            .select("id, name, ingredients, directions, ner") \
            .contains("ner", [ingredient.lower()]) \
            .limit(limit) \
            .execute()
        
        # If no exact match, try partial match
        if not result.data:
            result = supabase.table(TABLE_NAME) \
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
        
        result = supabase.table(TABLE_NAME) \
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
        result = supabase.table(TABLE_NAME) \
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


@app.route('/api/recipe/create', methods=['POST'])
def create_recipe():
    """Create a new recipe with GNN embedding."""
    data = request.get_json()
    
    required_fields = ['name', 'ingredients', 'directions', 'ner', 'graph_representation']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        # Generate GNN embedding from graph
        graph = data.get('graph_representation', {})
        embedding = generate_gnn_embedding(graph)
        
        recipe_data = {
            "name": data['name'],
            "ingredients": data['ingredients'],
            "directions": data['directions'],
            "ner": data['ner'],
            "graph_representation": graph,
            "tokens": data.get('tokens', []),
            "embedding": embedding,
            "link": data.get('link', ''),
            "source": data.get('source', 'user_submitted')
        }
        
        result = supabase.table(TABLE_NAME) \
            .insert(recipe_data) \
            .execute()
        
        if result.data:
            return jsonify({
                "message": "Recipe created successfully",
                "recipe": result.data[0],
                "embedding_generated": True
            }), 201
        else:
            return jsonify({"error": "Failed to create recipe"}), 500
    
    except Exception as e:
        print(f"Create recipe error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/similar', methods=['POST'])
def search_similar_recipes():
    """Find similar recipes using GNN embeddings."""
    data = request.get_json()
    
    recipe_id = data.get('recipe_id')
    graph = data.get('graph')
    top_k = data.get('top_k', 10)
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        if recipe_id:
            result = supabase.table(TABLE_NAME) \
                .select("embedding") \
                .eq("id", recipe_id) \
                .single() \
                .execute()
            
            if not result.data or not result.data.get('embedding'):
                return jsonify({"error": "Recipe not found or has no embedding"}), 404
            
            query_embedding = result.data['embedding']
        elif graph:
            query_embedding = generate_gnn_embedding(graph)
        else:
            return jsonify({"error": "Either recipe_id or graph is required"}), 400
        
        result = supabase.rpc(
            "search_recipes_gnn",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": top_k
            }
        ).execute()
        
        results = []
        for row in result.data:
            results.append({
                "id": row["id"],
                "name": row["name"],
                "ingredients": row.get("ingredients", []),
                "directions": row.get("directions", []),
                "ner": row.get("ner", []),
                "score": row.get("similarity", 0),
                "category": categorize_recipe(row.get("ner", []))
            })
        
        return jsonify({
            "results": results,
            "count": len(results),
            "message": "Similar recipes found"
        })
    
    except Exception as e:
        print(f"Similar search error: {e}")
        return jsonify({"error": str(e)}), 500


# ============== Startup ==============

# Load models when app starts
with app.app_context():
    print("🚀 Starting Flask app...")
    load_models()


@app.route('/api/recipe', methods=['POST'])
def create_recipe_from_steps():
    """Create recipe from name and steps, extracting all fields via OpenAI."""
    data = request.get_json()
    name = data.get('name', '').strip()
    steps = data.get('steps', '').strip()
    
    if not name or not steps:
        return jsonify({"error": "Both 'name' and 'steps' are required"}), 400
    
    try:
        # Extract structured data using OpenAI functions
        ingredients = extract_ingredients(steps)
        directions = extract_directions(steps)
        graph_representation = extract_graph_representation(steps)
        tokens = extract_tokens(steps)
        
        # Extract NER (simplified ingredient names) from ingredients
        ner = [ing.lower().split()[-1] for ing in ingredients] if ingredients else []
        
        # Generate GNN embedding from the graph
        embedding = generate_gnn_embedding(graph_representation)
        
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database not configured"}), 500
        
        recipe_data = {
            "name": name,
            "ingredients": ingredients,
            "directions": directions,
            "ner": ner,
            "graph_representation": graph_representation,
            "tokens": tokens,
            "embedding": embedding,
            "source": "user_submitted"
        }
        
        result = supabase.table(TABLE_NAME) \
            .insert(recipe_data) \
            .execute()
        
        if not result.data:
            return jsonify({"error": "Failed to create recipe"}), 500
        
        recipe = result.data[0]
        return jsonify({
            "message": "Recipe created successfully",
            "id": recipe["id"],
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
            "ner": recipe.get("ner", []),
            "graph": recipe.get("graph_representation", {}),
            "tokens": recipe.get("tokens", []),
            "embedding_generated": True
        }), 201
        
    except Exception as e:
        print(f"Create recipe error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':        
    app.run(debug=True, port=5000)
