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
import bcrypt
import base64
import zlib

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
    
    # Build node ID mapping (handle non-sequential or 1-indexed IDs)
    node_id_to_idx = {}
    for idx, node in enumerate(nodes):
        original_id = node.get("id", idx)
        node_id_to_idx[original_id] = idx
    
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
    num_nodes = x.size(0)
    
    # Build edge indices with ID remapping
    edge_list = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is not None and target is not None:
            # Remap to sequential indices
            source_idx = node_id_to_idx.get(source)
            target_idx = node_id_to_idx.get(target)
            
            # Skip invalid edges
            if source_idx is None or target_idx is None:
                continue
            if source_idx >= num_nodes or target_idx >= num_nodes:
                continue
            
            edge_list.append([source_idx, target_idx])
    
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
        
        # Validate edge_index before running through model
        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0:
            max_idx = data.edge_index.max().item()
            if max_idx >= num_nodes:
                print(f"Warning: edge_index has invalid indices (max={max_idx}, nodes={num_nodes}). Filtering edges.")
                # Filter out invalid edges
                valid_mask = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
                data.edge_index = data.edge_index[:, valid_mask]
        
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
        
        # Get all recipe names - paginate to overcome Supabase 1000 row limit
        all_names = []
        page_size = 1000
        offset = 0
        
        while True:
            result = supabase.table(TABLE_NAME) \
                .select("name") \
                .order("name") \
                .range(offset, offset + page_size - 1) \
                .execute()
            
            if not result.data:
                break
            
            all_names.extend([row["name"] for row in result.data])
            
            # If we got fewer than page_size, we've reached the end
            if len(result.data) < page_size:
                break
            
            offset += page_size
        
        names = all_names
        
        # Generate a version hash based on the names list
        # This will change when items are added/removed/modified
        version = hashlib.md5("".join(names).encode()).hexdigest()[:12]
        
        response = jsonify({
            "names": names,
            "version": version,
            "count": len(names)
        })
        # Prevent browser caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
        
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
def create_recipe_1():
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


@app.route('/api/recipe/generate', methods=['POST'])
def generate_recipe():
    """Generate recipe steps from a recipe name using AI."""
    data = request.get_json()
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"error": "Recipe name is required"}), 400
    
    try:
        steps = generate_recipe_steps(name)
        return jsonify({
            "name": name,
            "steps": steps,
            "message": "Recipe steps generated successfully"
        })
    except Exception as e:
        print(f"Generate recipe error: {e}")
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

def hash_password(password):
    """
    Hash a password using bcrypt.
    
    Args:
        password (str): Plain text password
    
    Returns:
        str: Hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password, hashed_password):
    """
    Verify a password against its hash.
    
    Args:
        plain_password (str): Plain text password
        hashed_password (str): Stored hashed password
    
    Returns:
        bool: True if password matches, False otherwise
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


@app.route('/auth/signup', methods=['POST'])
def signup():
    """Register a new user account."""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Validate input
        if not username:
            return jsonify({"error": "Username is required"}), 400
        
        if not password:
            return jsonify({"error": "Password is required"}), 400
        
        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Check if username already exists
        existing_user = supabase.table("users") \
            .select("username") \
            .eq("username", username) \
            .execute()
        
        if existing_user.data and len(existing_user.data) > 0:
            return jsonify({"error": "Username already exists"}), 409
        
        # Hash the password
        password_hash = hash_password(password)
        
        # Create user data
        user_data = {
            "username": username,
            "password_hash": password_hash
        }
        
        # Insert into database
        result = supabase.table("users").insert(user_data).execute()
        
        if not result.data:
            return jsonify({"error": "Failed to create user"}), 500
        
        print(f"User created successfully: {username}")
        
        return jsonify({
            "message": "User created successfully",
            "username": username
        }), 201
        
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500


@app.route('/auth/login', methods=['POST'])
def login():
    """Log in to user account."""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Validate input
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get user from database
        result = supabase.table("users") \
            .select("username, password_hash") \
            .eq("username", username) \
            .execute()
        
        # Check if user exists
        if not result.data or len(result.data) == 0:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Get stored password hash
        user = result.data[0]
        stored_hash = user['password_hash']
        
        # Verify password
        if verify_password(password, stored_hash):
            print(f"Login successful: {username}")
            return jsonify({
                "message": "Login successful",
                "username": username
            }), 200
        else:
            print(f"Login failed: {username} (incorrect password)")
            return jsonify({"error": "Invalid credentials"}), 401
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

@app.route('/auth/logout', methods=['POST'])
def logout():
    """Log out user (client-side should clear session/token)."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip() if data else None
        
        # In a stateless API, logout is primarily client-side
        # Server can log the event or invalidate tokens if using JWT
        
        if username:
            print(f"Logout: {username}")
            return jsonify({
                "message": "Logged out successfully",
                "username": username
            }), 200
        else:
            return jsonify({"message": "Logged out successfully"}), 200
        
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({"error": f"Logout failed: {str(e)}"}), 500


# ============== Palate Profile Functions ==============

# Dish names used in onboarding - for embedding lookup
ONBOARDING_DISH_NAMES = [
    '"Refried" Beans', "3 Bean Salad", "20 minute seared strip steak with sweet-and-sour carrots",
    "Almond Shortbread", "Almond Crescent", "5-Minute Fudge", "1950'S Potato Chip Cookies",
    "(Web Exclusive) Round 2 Recipe: Edamame with Pasta", "*Sweet And Sour Carrots",
    '"Pecan Pie" Acorn Squash'
]


def get_dish_embeddings_from_db(dish_names: list) -> dict:
    """Fetch embeddings for dishes from the database."""
    supabase = get_supabase_client()
    if not supabase:
        return {}
    
    embeddings = {}
    for name in dish_names:
        try:
            # Try exact match first
            result = supabase.table(TABLE_NAME) \
                .select("name, embedding") \
                .eq("name", name) \
                .limit(1) \
                .execute()
            
            if result.data and result.data[0].get("embedding"):
                emb = result.data[0]["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                embeddings[name] = emb
        except Exception as e:
            print(f"Error fetching embedding for {name}: {e}")
    
    return embeddings


def compute_palate_embedding(ratings: dict, dishes: list) -> list:
    """
    Compute a palate embedding as a weighted average of dish embeddings.
    Dishes rated higher have more influence on the palate.
    """
    # Create dish ID to name mapping
    dish_id_to_name = {str(d["id"]): d["name"] for d in dishes}
    
    # Get dish names that were rated
    rated_dish_names = [dish_id_to_name[str(did)] for did in ratings.keys() if str(did) in dish_id_to_name]
    
    # Fetch embeddings from database
    dish_embeddings = get_dish_embeddings_from_db(rated_dish_names)
    
    if not dish_embeddings:
        # Fallback: generate embeddings from dish names using sentence encoder
        if _sentence_encoder:
            weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            total_weight = 0
            for dish_id, rating in ratings.items():
                dish = dish_id_to_name.get(str(dish_id))
                if dish:
                    # Weight: higher ratings = more influence (rating 1-5 maps to 0.2-1.0)
                    weight = rating / 5.0
                    # Generate embedding from name
                    emb = _sentence_encoder.encode(dish, convert_to_numpy=True)
                    # Pad or truncate to EMBEDDING_DIM
                    if len(emb) < EMBEDDING_DIM:
                        emb = np.pad(emb, (0, EMBEDDING_DIM - len(emb)))
                    else:
                        emb = emb[:EMBEDDING_DIM]
                    weighted_sum += weight * emb
                    total_weight += weight
            
            if total_weight > 0:
                palate_emb = weighted_sum / total_weight
                # Normalize
                norm = np.linalg.norm(palate_emb)
                if norm > 0:
                    palate_emb = palate_emb / norm
                return palate_emb.tolist()
        return []
    
    # Compute weighted average of embeddings
    weighted_sum = None
    total_weight = 0
    
    for dish_id, rating in ratings.items():
        dish_name = dish_id_to_name.get(str(dish_id))
        if dish_name and dish_name in dish_embeddings:
            emb = np.array(dish_embeddings[dish_name], dtype=np.float32)
            # Weight: higher ratings = more influence
            weight = rating / 5.0
            
            if weighted_sum is None:
                weighted_sum = np.zeros_like(emb)
            
            weighted_sum += weight * emb
            total_weight += weight
    
    if weighted_sum is not None and total_weight > 0:
        palate_emb = weighted_sum / total_weight
        # Normalize
        norm = np.linalg.norm(palate_emb)
        if norm > 0:
            palate_emb = palate_emb / norm
        return palate_emb.tolist()
    
    return []


def encode_palate(ratings: dict, categories: dict, embedding: list = None) -> str:
    """
    Encode palate ratings into a shareable string.
    
    Format: base64(compressed(json))
    """
    try:
        # Build palate data
        palate_data = {
            "r": ratings,  # ratings: {dish_id: rating}
            "c": categories,  # category preferences computed from ratings
            "v": 1  # version for future compatibility
        }
        
        # Add embedding if provided (truncate for size)
        if embedding:
            # Store first 64 dims to keep code size reasonable
            palate_data["e"] = [round(x, 4) for x in embedding[:64]]
        
        # Convert to JSON, compress, and encode
        json_str = json.dumps(palate_data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode('utf-8'))
        encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
        
        # Remove padding for cleaner code
        encoded = encoded.rstrip('=')
        
        return encoded
    except Exception as e:
        print(f"Error encoding palate: {e}")
        return ""


def decode_palate(palate_code: str) -> dict:
    """
    Decode a palate code back into ratings and preferences.
    """
    try:
        # Add back padding if needed
        padding = 4 - (len(palate_code) % 4)
        if padding != 4:
            palate_code += '=' * padding
        
        # Decode and decompress
        compressed = base64.urlsafe_b64decode(palate_code.encode('utf-8'))
        json_str = zlib.decompress(compressed).decode('utf-8')
        palate_data = json.loads(json_str)
        
        return palate_data
    except Exception as e:
        print(f"Error decoding palate: {e}")
        return None


def compute_category_preferences(ratings: dict, dishes: list) -> dict:
    """
    Compute category preferences from individual dish ratings.
    """
    category_scores = {}
    category_counts = {}
    
    # Create dish lookup
    dish_lookup = {str(d["id"]): d for d in dishes}
    
    for dish_id, rating in ratings.items():
        dish = dish_lookup.get(str(dish_id))
        if dish:
            category = dish.get("category", "Other")
            if category not in category_scores:
                category_scores[category] = 0
                category_counts[category] = 0
            category_scores[category] += rating
            category_counts[category] += 1
    
    # Calculate averages
    preferences = {}
    for category, total_score in category_scores.items():
        count = category_counts[category]
        preferences[category] = round(total_score / count, 2) if count > 0 else 0
    
    return preferences


@app.route('/api/palate/create', methods=['POST'])
def create_palate():
    """Create a palate profile from dish ratings."""
    data = request.get_json()
    
    username = data.get('username', '').strip()
    ratings = data.get('ratings', {})
    dishes = data.get('dishes', [])
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    if not ratings or len(ratings) < 5:
        return jsonify({"error": "At least 5 dish ratings are required"}), 400
    
    try:
        # Compute category preferences
        category_prefs = compute_category_preferences(ratings, dishes)
        
        # Compute palate embedding from dish embeddings
        palate_embedding = compute_palate_embedding(ratings, dishes)
        
        # Generate palate code with embedding
        palate_code = encode_palate(ratings, category_prefs, palate_embedding)
        
        if not palate_code:
            return jsonify({"error": "Failed to generate palate code"}), 500
        
        # Store in database
        supabase = get_supabase_client()
        if supabase:
            # Update or insert user palate
            result = supabase.table("users") \
                .update({
                    "palate_code": palate_code,
                    "palate_ratings": ratings,
                    "palate_categories": category_prefs,
                    "is_onboarded": True
                }) \
                .eq("username", username) \
                .execute()
            
            if not result.data:
                print(f"Warning: Could not update user palate for {username}")
        
        return jsonify({
            "message": "Palate profile created successfully",
            "palate_code": palate_code,
            "category_preferences": category_prefs
        })
        
    except Exception as e:
        print(f"Create palate error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/palate/import', methods=['POST'])
def import_palate():
    """Import a palate profile from a code."""
    data = request.get_json()
    
    username = data.get('username', '').strip()
    palate_code = data.get('palate_code', '').strip()
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    if not palate_code:
        return jsonify({"error": "Palate code is required"}), 400
    
    try:
        # Decode the palate
        palate_data = decode_palate(palate_code)
        
        if not palate_data:
            return jsonify({"error": "Invalid palate code"}), 400
        
        ratings = palate_data.get("r", {})
        categories = palate_data.get("c", {})
        
        # Store in database
        supabase = get_supabase_client()
        if supabase:
            result = supabase.table("users") \
                .update({
                    "palate_code": palate_code,
                    "palate_ratings": ratings,
                    "palate_categories": categories,
                    "is_onboarded": True
                }) \
                .eq("username", username) \
                .execute()
            
            if not result.data:
                return jsonify({"error": "User not found"}), 404
        
        return jsonify({
            "message": "Palate imported successfully",
            "palate_code": palate_code,
            "category_preferences": categories
        })
        
    except Exception as e:
        print(f"Import palate error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/palate/check', methods=['GET'])
def check_palate():
    """Check if a user has been onboarded."""
    username = request.args.get('username', '').strip()
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"is_onboarded": False})
        
        result = supabase.table("users") \
            .select("is_onboarded, palate_code, palate_categories") \
            .eq("username", username) \
            .single() \
            .execute()
        
        if not result.data:
            return jsonify({"is_onboarded": False})
        
        return jsonify({
            "is_onboarded": result.data.get("is_onboarded", False),
            "palate_code": result.data.get("palate_code", ""),
            "category_preferences": result.data.get("palate_categories", {})
        })
        
    except Exception as e:
        print(f"Check palate error: {e}")
        return jsonify({"is_onboarded": False})


@app.route('/api/palate/decode', methods=['POST'])
def decode_palate_endpoint():
    """Decode a palate code to see preferences (public)."""
    data = request.get_json()
    palate_code = data.get('palate_code', '').strip()
    
    if not palate_code:
        return jsonify({"error": "Palate code is required"}), 400
    
    try:
        palate_data = decode_palate(palate_code)
        
        if not palate_data:
            return jsonify({"error": "Invalid palate code"}), 400
        
        return jsonify({
            "ratings": palate_data.get("r", {}),
            "category_preferences": palate_data.get("c", {}),
            "version": palate_data.get("v", 1)
        })
        
    except Exception as e:
        print(f"Decode palate error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/palate/compare', methods=['POST'])
def compare_palates():
    """Compare two palate codes and return similarity analysis."""
    data = request.get_json()
    # Accept both naming conventions
    code1 = (data.get('code1') or data.get('palate_code_1') or '').strip()
    code2 = (data.get('code2') or data.get('palate_code_2') or '').strip()
    
    if not code1 or not code2:
        return jsonify({"error": "Both palate codes are required"}), 400
    
    try:
        palate1 = decode_palate(code1)
        palate2 = decode_palate(code2)
        
        if not palate1 or not palate2:
            return jsonify({"error": "Invalid palate code(s)"}), 400
        
        ratings1 = palate1.get("r", {})
        ratings2 = palate2.get("r", {})
        cats1 = palate1.get("c", {})
        cats2 = palate2.get("c", {})
        emb1 = palate1.get("e", [])
        emb2 = palate2.get("e", [])
        
        # Calculate embedding similarity (primary metric)
        embedding_similarity = 0.0
        if emb1 and emb2:
            embedding_similarity = cosine_similarity(emb1, emb2)
            embedding_similarity = max(0, min(1, embedding_similarity))  # Clamp to [0,1]
        
        # Calculate rating similarity (dishes rated by both)
        common_dishes = set(ratings1.keys()) & set(ratings2.keys())
        if common_dishes:
            rating_diff = sum(abs(float(ratings1[d]) - float(ratings2[d])) for d in common_dishes)
            max_diff = len(common_dishes) * 4  # Max difference is 4 per dish (1-5 scale)
            rating_similarity = 1 - (rating_diff / max_diff) if max_diff > 0 else 0
        else:
            rating_similarity = 0.5  # Default if no common dishes
        
        # Calculate category preference similarity
        all_cats = set(cats1.keys()) | set(cats2.keys())
        if all_cats:
            cat_diff = sum(abs(float(cats1.get(c, 2.5)) - float(cats2.get(c, 2.5))) for c in all_cats)
            max_cat_diff = len(all_cats) * 4  # Max difference per category (1-5 scale)
            cat_similarity = 1 - (cat_diff / max_cat_diff) if max_cat_diff > 0 else 0
        else:
            cat_similarity = 0.5
        
        # Overall similarity - prioritize embedding if available
        if emb1 and emb2:
            overall_similarity = (embedding_similarity * 0.5 + rating_similarity * 0.3 + cat_similarity * 0.2)
        else:
            overall_similarity = (rating_similarity * 0.6 + cat_similarity * 0.4)
        
        # Ensure valid number
        overall_similarity = max(0, min(1, overall_similarity))
        
        # Find common liked categories (both have preference > 2.5)
        common_liked = [c for c in all_cats if float(cats1.get(c, 0)) > 2.5 and float(cats2.get(c, 0)) > 2.5]
        
        # Find different preferences (one likes, other doesn't)
        different_prefs = [c for c in all_cats 
                          if (float(cats1.get(c, 0)) > 3 and float(cats2.get(c, 0)) < 2.5) or 
                             (float(cats2.get(c, 0)) > 3 and float(cats1.get(c, 0)) < 2.5)]
        
        return jsonify({
            "similarity": round(overall_similarity, 3),
            "overall_similarity": round(overall_similarity * 100, 1),
            "rating_similarity": round(rating_similarity * 100, 1),
            "category_similarity": round(cat_similarity * 100, 1),
            "embedding_similarity": round(embedding_similarity * 100, 1) if emb1 and emb2 else None,
            "common_dishes_rated": len(common_dishes),
            "common_categories": common_liked,
            "common_liked_categories": common_liked,
            "different_categories": different_prefs,
            "different_preferences": different_prefs
        })
        
    except Exception as e:
        print(f"Compare palates error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/palate/intersection', methods=['POST'])
def find_palate_intersection():
    """Find recipes that both users would likely enjoy based on their palates."""
    data = request.get_json()
    # Accept both naming conventions
    code1 = (data.get('code1') or data.get('palate_code_1') or '').strip()
    code2 = (data.get('code2') or data.get('palate_code_2') or '').strip()
    limit = data.get('limit', 10)
    
    if not code1 or not code2:
        return jsonify({"error": "Both palate codes are required"}), 400
    
    try:
        palate1 = decode_palate(code1)
        palate2 = decode_palate(code2)
        
        if not palate1 or not palate2:
            return jsonify({"error": "Invalid palate code(s)"}), 400
        
        cats1 = palate1.get("c", {})
        cats2 = palate2.get("c", {})
        
        # Find categories both users like (preference > 2.5)
        common_liked = [c for c in set(cats1.keys()) & set(cats2.keys()) 
                        if cats1.get(c, 0) > 2.5 and cats2.get(c, 0) > 2.5]
        
        if not common_liked:
            return jsonify({
                "recipes": [],
                "message": "No common preferred categories found"
            })
        
        # Query recipes that match common liked categories
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get palate embeddings for similarity calculation
        emb1 = palate1.get("e", [])
        emb2 = palate2.get("e", [])
        
        # Compute combined palate embedding (average of both)
        combined_embedding = None
        if emb1 and emb2:
            combined_embedding = [(a + b) / 2 for a, b in zip(emb1, emb2)]
        elif emb1:
            combined_embedding = emb1
        elif emb2:
            combined_embedding = emb2
        
        # Fetch recipes with embeddings to compute actual similarity
        recipes = []
        response = supabase.table(TABLE_NAME)\
            .select("name, embedding, ner")\
            .not_.is_("embedding", "null")\
            .limit(100)\
            .execute()
        
        if response.data:
            for r in response.data:
                recipe_embedding = r.get("embedding")
                if recipe_embedding:
                    if isinstance(recipe_embedding, str):
                        recipe_embedding = json.loads(recipe_embedding)
                    
                    # Calculate similarity score
                    if combined_embedding and recipe_embedding:
                        # Truncate recipe embedding to match palate embedding size (64 dims)
                        recipe_emb_truncated = recipe_embedding[:len(combined_embedding)]
                        score = cosine_similarity(combined_embedding, recipe_emb_truncated)
                        score = max(0, min(1, score))  # Clamp to [0,1]
                    else:
                        score = 0.5  # Default if no embeddings
                    
                    # Determine category from NER if available
                    ner = r.get("ner", [])
                    category = "Recipe"
                    if ner:
                        # Check if any common_liked category appears in ingredients
                        for cat in common_liked:
                            if any(cat.lower() in str(ing).lower() for ing in ner):
                                category = cat
                                break
                    
                    recipes.append({
                        "name": r.get("name", "Unknown"),
                        "category": category,
                        "score": score
                    })
        
        # Sort by similarity score (highest first) and limit
        recipes.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates and limit
        seen = set()
        unique_recipes = []
        for r in recipes:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_recipes.append({
                    "name": r["name"],
                    "category": r.get("category", "Recipe"),
                    "score": round(r.get("score", 0.5), 3)
                })
                if len(unique_recipes) >= limit:
                    break
        
        # If no exact name matches, get random recipes as fallback
        if not unique_recipes:
            response = supabase.table(TABLE_NAME)\
                .select("name")\
                .limit(limit)\
                .execute()
            if response.data:
                for r in response.data:
                    unique_recipes.append({
                        "name": r.get("name", "Unknown"),
                        "category": "Recommended",
                        "score": 0.7
                    })
        
        return jsonify({
            "recipes": unique_recipes,
            "common_categories": common_liked,
            "total_found": len(unique_recipes)
        })
        
    except Exception as e:
        print(f"Intersection error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':        
    app.run(host='0.0.0.0', debug=True, port=5000)
