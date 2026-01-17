"""
Train a real Graph Neural Network for recipe embeddings.
Requires: torch, torch_geometric, sentence-transformers
"""

import os
import json
import hashlib
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "recipes_duplicate_GNN2"

# Model hyperparameters (optimized)
EMBEDDING_DIM = 512  # Keep 512 for database compatibility
HIDDEN_DIM = 256
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Lower LR for stability

# Node feature dimension - using sentence transformer output
NODE_FEATURE_DIM = 64  # Will be set based on embedding model


class RecipeGAT(nn.Module):
    """Graph Attention Network for recipe embeddings - better than GCN."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Multi-head attention for better message passing
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, edge_index, batch):
        # Graph attention convolutions with residual connections
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        return x


# Keep old GCN as fallback
class RecipeGNN(nn.Module):
    """Graph Neural Network for recipe embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


def graph_to_pyg_data(graph: dict, ingredient_embeddings: dict, embed_dim: int) -> Data:
    """Convert recipe graph to PyTorch Geometric Data object."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    # Node features
    node_features = []
    for node in nodes:
        if node["type"] == "ingredient":
            # Use pre-computed ingredient embedding
            label = node["label"].lower()
            feat = ingredient_embeddings.get(label, np.zeros(embed_dim, dtype=np.float32))
        else:
            # Learned embeddings for intermediate/final nodes
            feat = np.zeros(embed_dim, dtype=np.float32)
            if node["type"] == "intermediate":
                feat[0] = 1.0
                feat[1] = 0.5
            elif node["type"] == "final":
                feat[0] = 0.5
                feat[1] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Edge indices
    edge_index = []
    for edge in edges:
        edge_index.append([edge["source"], edge["target"]])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


def compute_ingredient_similarity(ner1: list, ner2: list) -> float:
    """Compute Jaccard similarity between ingredient lists."""
    if not ner1 or not ner2:
        return 0.0
    set1 = set(ing.lower().strip() for ing in ner1 if isinstance(ing, str))
    set2 = set(ing.lower().strip() for ing in ner2 if isinstance(ing, str))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def train_gnn(recipes: list[dict], ingredient_embeddings: dict, embed_dim: int):
    """Train the GNN model with supervised contrastive loss."""
    print("🔧 Converting graphs to PyG format...")
    
    # Debug: Print first recipe's graph_representation
    if recipes:
        first_graph = recipes[0].get("graph_representation")
        print(f"   DEBUG - First graph type: {type(first_graph)}")
        print(f"   DEBUG - First graph preview: {str(first_graph)[:200] if first_graph else 'None'}...")
    
    # Convert all graphs
    data_list = []
    recipe_ners = []  # Store NER for supervised loss
    skipped_no_graph = 0
    skipped_no_nodes = 0
    skipped_error = 0
    first_error = None
    
    for recipe in tqdm(recipes, desc="Converting"):
        graph = recipe.get("graph_representation")
        
        # Handle JSON string if needed
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except json.JSONDecodeError:
                skipped_error += 1
                continue
        
        if not graph:
            skipped_no_graph += 1
            continue
            
        if not isinstance(graph, dict) or not graph.get("nodes"):
            skipped_no_nodes += 1
            continue
            
        try:
            data = graph_to_pyg_data(graph, ingredient_embeddings, embed_dim)
            data.recipe_id = recipe["id"]
            data_list.append(data)
            
            # Store NER for computing ingredient similarity
            ner = recipe.get("ner", [])
            if isinstance(ner, str):
                try:
                    ner = json.loads(ner)
                except:
                    ner = []
            recipe_ners.append(ner if isinstance(ner, list) else [])
        except Exception as e:
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
            skipped_error += 1
            continue
    
    print(f"   Skipped - no graph: {skipped_no_graph}, no nodes: {skipped_no_nodes}, errors: {skipped_error}")
    if first_error:
        print(f"   First error: {first_error}")
    
    print(f"✅ Converted {len(data_list)} graphs")
    
    if len(data_list) < 10:
        print("❌ Not enough valid graphs to train")
        return None, None
    
    # Create data loader with indices for ingredient similarity lookup
    # Store NER mapping by index
    for i, data in enumerate(data_list):
        data.ner_idx = i
    
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model - use GAT for better performance
    input_dim = embed_dim
    model = RecipeGAT(input_dim, HIDDEN_DIM, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Supervised contrastive loss using ingredient similarity
    def supervised_contrastive_loss(embeddings, batch_ner_indices, temperature=0.5):
        """Contrastive loss weighted by ingredient similarity."""
        n = embeddings.size(0)
        if n < 2:
            return torch.tensor(0.0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute embedding similarity matrix
        emb_sim = torch.mm(embeddings, embeddings.t()) / temperature
        
        # Compute ingredient similarity matrix (ground truth)
        ing_sim = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ing_sim[i, j] = 1.0
                else:
                    idx_i = batch_ner_indices[i].item()
                    idx_j = batch_ner_indices[j].item()
                    ing_sim[i, j] = compute_ingredient_similarity(
                        recipe_ners[idx_i], recipe_ners[idx_j]
                    )
        
        # Combined loss: pull similar recipes together, push different apart
        # 1. InfoNCE component (self-supervised)
        labels = torch.arange(n, device=embeddings.device)
        infonce_loss = F.cross_entropy(emb_sim, labels)
        
        # 2. Similarity alignment component (supervised)
        # Scale ingredient similarity to match logit scale
        target_sim = ing_sim * 2 - 1  # Scale to [-1, 1]
        emb_sim_normalized = torch.tanh(emb_sim * temperature)
        alignment_loss = F.mse_loss(emb_sim_normalized, target_sim)
        
        # Combined loss with weighting
        loss = 0.5 * infonce_loss + 0.5 * alignment_loss
        return loss
    
    print(f"\n🚀 Training GAT for {NUM_EPOCHS} epochs...")
    
    model.train()
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(batch.x, batch.edge_index, batch.batch)
            
            # Compute supervised contrastive loss
            loss = supervised_contrastive_loss(embeddings, batch.ner_idx)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
    
    print("✅ Training complete!")
    return model, data_list


def generate_embeddings(model, data_list) -> dict:
    """Generate embeddings for all recipes."""
    print("\n📊 Generating embeddings...")
    
    model.eval()
    embeddings = {}
    
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            emb = model(batch.x, batch.edge_index, batch.batch)
            
            # Get individual recipe embeddings
            for i, recipe_id in enumerate(batch.recipe_id):
                embeddings[recipe_id] = emb[i].numpy().tolist()
    
    return embeddings


def build_ingredient_embeddings(recipes: list[dict]) -> tuple[dict, int]:
    """Build semantic embeddings for ingredients using sentence-transformers."""
    
    all_ingredients = set()
    for recipe in recipes:
        ner = recipe.get("ner", [])
        if isinstance(ner, list):
            for ing in ner:
                if isinstance(ing, str):
                    all_ingredients.add(ing.lower())
    
    print(f"   Found {len(all_ingredients)} unique ingredients")
    
    # Try to use sentence-transformers for semantic embeddings
    try:
        from sentence_transformers import SentenceTransformer
        print("   Using sentence-transformers for semantic embeddings...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        ingredients_list = list(all_ingredients)
        
        # Batch encode all ingredients
        print("   Encoding ingredients...")
        embeddings_array = model.encode(
            ingredients_list, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Use first 64 dimensions for efficiency
        embed_dim = min(64, embeddings_array.shape[1])
        embeddings = {}
        for i, ing in enumerate(ingredients_list):
            embeddings[ing] = embeddings_array[i, :embed_dim].astype(np.float32)
        
        print(f"✅ Created semantic embeddings for {len(embeddings)} ingredients (dim={embed_dim})")
        return embeddings, embed_dim
        
    except ImportError:
        print("   ⚠️ sentence-transformers not installed, falling back to hash embeddings")
        print("   Install with: pip install sentence-transformers")
        
        # Fallback to hash-based embeddings
        embed_dim = 32
        embeddings = {}
        for ing in all_ingredients:
            hash_bytes = hashlib.sha256(ing.encode()).digest()
            feat = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            feat = (feat - 128) / 128  # Normalize to [-1, 1]
            embeddings[ing] = feat
        
        print(f"✅ Created hash embeddings for {len(embeddings)} ingredients (dim={embed_dim})")
        return embeddings, embed_dim


def main():
    print("=" * 60)
    print("GNN EMBEDDING TRAINING")
    print("=" * 60)
    
    # Check for PyTorch Geometric
    try:
        import torch_geometric
        print(f"✅ PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("❌ PyTorch Geometric not installed!")
        print("   Install with: pip install torch-geometric")
        return
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Fetch recipes
    print("\n📂 Fetching recipes...")
    recipes = []
    offset = 0
    while True:
        response = supabase.table(TABLE_NAME).select(
            "id, name, ner, graph_representation"
        ).range(offset, offset + 999).execute()
        
        if not response.data:
            break
        recipes.extend(response.data)
        offset += 1000
    
    print(f"✅ Fetched {len(recipes)} recipes")
    
    # Build semantic ingredient embeddings
    ingredient_embeddings, embed_dim = build_ingredient_embeddings(recipes)
    
    # Train GNN
    model, data_list = train_gnn(recipes, ingredient_embeddings, embed_dim)
    
    if model is None:
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(model, data_list)
    
    # Update database
    print(f"\n💾 Updating {len(embeddings)} recipes in database...")
    for recipe_id, embedding in tqdm(embeddings.items(), desc="Saving"):
        supabase.table(TABLE_NAME).update({
            "embedding": embedding
        }).eq("id", recipe_id).execute()
    
    print("\n✅ Done!")
    
    # After training loop ends, save the model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(model_dir, 'recipe_gnn.pt')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model to {model_path}")
    
    # Save model config for loading later
    config = {
        "input_dim": NODE_FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": EMBEDDING_DIM
    }
    config_path = os.path.join(model_dir, 'recipe_gnn_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"✅ Saved config to {config_path}")


if __name__ == "__main__":
    main()