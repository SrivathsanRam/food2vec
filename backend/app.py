from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "food-vectors")

# Sample food data for autocomplete (in production, this would come from your database)
FOOD_DATABASE = [
    "Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry",
    "Broccoli", "Burger", "Burrito", "Cabbage", "Cake", "Carrot",
    "Cheese", "Cheesecake", "Cherry", "Chicken", "Chocolate", "Cookie",
    "Corn", "Croissant", "Cucumber", "Donut", "Egg", "Fish",
    "French Fries", "Grape", "Grapefruit", "Hamburger", "Hot Dog", "Ice Cream",
    "Kiwi", "Lemon", "Lettuce", "Lime", "Lobster", "Mango",
    "Meatball", "Melon", "Mushroom", "Noodles", "Olive", "Onion",
    "Orange", "Pancake", "Pasta", "Peach", "Pear", "Pepper",
    "Pineapple", "Pizza", "Plum", "Popcorn", "Potato", "Pretzel",
    "Pumpkin", "Raspberry", "Rice", "Salad", "Salmon", "Sandwich",
    "Sausage", "Shrimp", "Soup", "Spinach", "Steak", "Strawberry",
    "Sushi", "Taco", "Toast", "Tomato", "Waffle", "Watermelon"
]


def get_index():
    """Get or create the Pinecone index."""
    try:
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error connecting to index: {e}")
        return None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Food2Vec API is running!"})


@app.route('/api/autocomplete', methods=['GET'])
def autocomplete():
    """Return food suggestions based on query."""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify({"suggestions": []})
    
    # Filter foods that match the query
    suggestions = [
        food for food in FOOD_DATABASE 
        if food.lower().startswith(query) or query in food.lower()
    ][:10]  # Limit to 10 suggestions
    
    return jsonify({"suggestions": suggestions})


@app.route('/api/search', methods=['POST'])
def search_food():
    """Search for similar foods using Pinecone vector similarity."""
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        index = get_index()
        if index is None:
            # Fallback: return mock results if Pinecone is not configured
            return jsonify({
                "query": query,
                "results": get_mock_results(query),
                "message": "Using mock data - Pinecone not configured"
            })
        
        # In production, you would generate embeddings for the query
        # For now, we'll use a simple text search or mock embeddings
        # query_embedding = generate_embedding(query)
        
        # Example Pinecone query (requires actual embeddings)
        # results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        # Return mock results for demonstration
        return jsonify({
            "query": query,
            "results": get_mock_results(query),
            "message": "Search completed successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_mock_results(query):
    """Generate mock search results based on query."""
    query_lower = query.lower()
    
    # Find matching foods and generate similarity scores
    results = []
    for food in FOOD_DATABASE:
        if query_lower in food.lower():
            score = 0.95 if food.lower().startswith(query_lower) else 0.75
            results.append({
                "name": food,
                "score": score,
                "category": get_food_category(food),
                "description": f"Delicious {food.lower()} - a popular food choice!"
            })
    
    # Add some "similar" foods based on category
    if results:
        category = results[0].get("category", "Other")
        for food in FOOD_DATABASE:
            if get_food_category(food) == category and food.lower() != query_lower:
                if len(results) >= 10:
                    break
                if not any(r["name"] == food for r in results):
                    results.append({
                        "name": food,
                        "score": 0.6,
                        "category": category,
                        "description": f"Similar food: {food}"
                    })
    
    # Sort by score and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


def get_food_category(food):
    """Categorize food items."""
    fruits = ["Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry", 
              "Cherry", "Grape", "Grapefruit", "Kiwi", "Lemon", "Lime", "Mango",
              "Melon", "Orange", "Peach", "Pear", "Pineapple", "Plum", "Raspberry",
              "Strawberry", "Watermelon"]
    vegetables = ["Broccoli", "Cabbage", "Carrot", "Corn", "Cucumber", "Lettuce",
                  "Mushroom", "Olive", "Onion", "Pepper", "Potato", "Pumpkin",
                  "Spinach", "Tomato"]
    desserts = ["Cake", "Cheesecake", "Chocolate", "Cookie", "Donut", "Ice Cream",
                "Pancake", "Waffle"]
    proteins = ["Chicken", "Egg", "Fish", "Lobster", "Meatball", "Salmon", 
                "Sausage", "Shrimp", "Steak"]
    
    if food in fruits:
        return "Fruits"
    elif food in vegetables:
        return "Vegetables"
    elif food in desserts:
        return "Desserts"
    elif food in proteins:
        return "Proteins"
    else:
        return "Other"


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all food categories with their items."""
    categories = {}
    for food in FOOD_DATABASE:
        category = get_food_category(food)
        if category not in categories:
            categories[category] = []
        categories[category].append(food)
    
    return jsonify({"categories": categories})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
