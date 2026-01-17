"""
Build semantic graph-based embeddings for recipes using:
1. Flavor categories (sweet, sour, umami, bitter, salty, fat, spicy)
2. Texture categories (crunchy, creamy, starchy)
3. Cooking action categories (heat_action)

Uses the same filter_dict from download_cleaned_data.ipynb for consistency.
"""

import os
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
FLAVOR_DIM = 64
ACTION_DIM = 32


# ============================================================================
# FLAVOR, TEXTURE & ACTION CATEGORIES (from download_cleaned_data.ipynb)
# ============================================================================

FILTER_DICT = {
    'sweet': [
        'sugar', 'honey', 'maple syrup', 'dates', 'agave', 'molasses', 'brown sugar', "confectioners' sugar", 'cane syrup',
        'stevia', 'monk fruit', 'coconut sugar', 'jaggery', 'corn syrup', 'rice syrup', 'barley malt', 'sucanat',
        'caramel', 'butterscotch', 'frosting', 'icing', 'fondant', 'jam', 'jelly', 'marmalade', 'preserves',
        'fruit compote', 'dulce de leche', 'sweetened condensed milk', 'apple sauce', 'mashed banana', 'ripe mango',
        'pineapple', 'berries', 'grapes', 'melon', 'peaches', 'plums', 'cherries', 'figs', 'raisins', 'dried apricots',
        'prunes', 'sweet potato', 'carrots', 'beets', 'corn', 'peas', 'bell peppers', 'onions (caramelized)',
        'balsamic glaze', 'hoisin sauce', 'teriyaki sauce', 'ketchup', 'barbecue sauce', 'sweet chili sauce',
        'vanilla', 'almond extract', 'coconut milk', 'sweetened yogurt', 'sweet cream', 'marzipan', 'nougat'
    ],
    'sour': [
        'vinegar', 'white vinegar', 'apple cider vinegar', 'red wine vinegar', 'balsamic vinegar', 'rice vinegar',
        'champagne vinegar', 'sherry vinegar', 'malt vinegar', 'distilled vinegar', 'lemon', 'lime', 'yogurt',
        'buttermilk', 'sour cream', 'crème fraîche', 'kefir', 'sauerkraut', 'kimchi', 'pickles', 'pickled vegetables',
        'tamarind', 'sumac', 'verjus', 'sorrel', 'rhubarb', 'green apples', 'green grapes', 'gooseberries',
        'cranberries', 'currants', 'passion fruit', 'tart cherries', 'unripe mango', 'green papaya', 'citrus zest',
        'fermented foods', 'kombucha', 'sourdough', 'sour candies', 'citric acid', 'tartaric acid', 'malic acid'
    ],
    'umami': [
        'soy sauce', 'tamari', 'coconut aminos', 'fish sauce', 'oyster sauce', 'worcestershire sauce', 'hoisin sauce',
        'miso', 'doubanjiang', 'gochujang', 'parmesan cheese', 'pecorino', 'aged gouda', 'blue cheese', 'nutritional yeast',
        'anchovy', 'sardines', 'mackerel', 'bonito flakes', 'dashi', 'kombu', 'nori', 'seaweed', 'mushrooms',
        'shiitake', 'porcini', 'morel', 'chanterelle', 'enoki', 'maitake', 'truffle', 'truffle oil', 'tomato paste',
        'sun-dried tomatoes', 'roasted tomatoes', 'caramelized onions', 'roasted garlic', 'marmite', 'vegemite',
        'bovril', 'beef stock', 'chicken stock', 'vegetable stock', 'bone broth', 'demiglace', 'gravy', 'browned meat',
        'dry-aged beef', 'cured meats', 'fermented beans', 'black garlic', 'soybean paste', 'msg', 'autolyzed yeast extract'
    ],
    'bitter': [
        'cocoa', 'dark chocolate', 'cacao nibs', 'coffee', 'espresso', 'espresso powder', 'matcha', 'green tea',
        'black tea', 'tonic water', 'quinine', 'angostura bitters', 'campari', 'aperol', 'vermouth', 'absinthe',
        'grapefruit', 'bitter melon', 'endive', 'escarole', 'radicchio', 'kale', 'collard greens', 'mustard greens',
        'arugula', 'watercress', 'dandelion greens', 'brussels sprouts', 'broccoli rabe', 'artichokes', 'asparagus',
        'eggplant', 'saffron', 'turmeric', 'fenugreek', 'szechuan peppercorn', 'citrus pith', 'almond skins',
        'walnuts', 'hazelnuts', 'pistachios', 'sesame seeds', 'burnt sugar', 'charred vegetables', 'smoked ingredients',
        'neem', 'gentian root', 'wormwood', 'cascara', 'hops'
    ],
    'salty': [
        'salt', 'sea salt', 'kosher salt', 'himalayan salt', 'flaky salt', 'smoked salt', 'garlic salt', 'celery salt',
        'soy sauce', 'tamari', 'liquid aminos', 'fish sauce', 'anchovy paste', 'capers', 'olives', 'green olives',
        'kalamata olives', 'castelvetrano olives', 'feta cheese', 'goat cheese', 'halloumi', 'queso fresco', 'cotija',
        'blue cheese', 'gorgonzola', 'pecorino', 'parmesan', 'aged cheddar', 'cured meats', 'bacon', 'pancetta',
        'guanciale', 'prosciutto', 'serrano ham', 'speck', 'salami', 'pepperoni', 'soppressata', 'chorizo', 'sausage',
        'salted butter', 'salted nuts', 'pretzels', 'crackers', 'potato chips', 'popcorn', 'pickles', 'kimchi',
        'sauerkraut', 'fermented vegetables', 'bouillon', 'stock cubes', 'miso paste', 'oyster sauce', 'worcestershire sauce',
        'teriyaki sauce', 'hoisin sauce', 'salted egg yolk', 'salt-packed sardines', 'salt cod', 'biltong', 'jerky'
    ],
    'fat': [
        'butter', 'unsalted butter', 'clarified butter', 'ghee', 'brown butter', 'compound butter',
        'cream', 'heavy cream', 'whipping cream', 'light cream', 'half-and-half', 'sour cream', 'crème fraîche',
        'mascarpone', 'cream cheese', 'neufchâtel', 'oil', 'olive oil', 'extra virgin olive oil', 'vegetable oil',
        'canola oil', 'grapeseed oil', 'avocado oil', 'coconut oil', 'peanut oil', 'sesame oil', 'truffle oil',
        'walnut oil', 'almond oil', 'sunflower oil', 'corn oil', 'lard', 'schmaltz', 'duck fat', 'goose fat',
        'beef tallow', 'suet', 'bacon fat', 'rendered fat', 'cheese', 'hard cheeses', 'soft cheeses', 'aged cheeses',
        'fresh cheeses', 'avocado', 'nuts', 'nut butters', 'seeds', 'seed butters', 'egg yolk', 'foie gras',
        'fatty fish', 'marbled meat', 'shortening', 'margarine', 'copha', 'mayonnaise', 'aioli', 'ranch dressing',
        'caesar dressing', 'tahini', 'chocolate', 'cocoa butter'
    ],
    'spicy': [
        'chili', 'chili pepper', 'jalapeño', 'serrano', 'habanero', 'scotch bonnet', 'ghost pepper', 'carolina reaper',
        'cayenne pepper', 'red pepper flakes', 'crushed red pepper', 'chili powder', 'ancho chili', 'chipotle',
        'guajillo', 'pasilla', 'arbol', 'piri piri', "bird's eye chili", 'thai chili', 'szechuan pepper',
        'black pepper', 'white pepper', 'green pepper', 'pink pepper', 'long pepper', 'tellicherry pepper',
        'wasabi', 'horseradish', 'mustard', 'dijon mustard', 'english mustard', 'whole grain mustard',
        'hot mustard', 'mustard seeds', 'ginger', 'fresh ginger', 'pickled ginger', 'galangal', 'turmeric',
        'horseradish root', 'radish', 'daikon', 'watercress', 'arugula', 'garlic', 'raw garlic', 'fermented garlic',
        'onion', 'raw onion', 'shallot', 'leek', 'chives', 'hot sauce', 'tabasco', 'sriracha', 'sambal',
        'harissa', 'gochujang', 'zhoug', 'chermoula', 'peri peri sauce', 'buffalo sauce', 'nashville hot sauce',
        'cajun seasoning', 'creole seasoning', 'berbere', 'ras el hanout', 'curry powder', 'curry paste'
    ],
    'heat_action': [
        'saute', 'sauté', 'pan-fry', 'shallow fry', 'deep fry', 'air fry', 'stir-fry', 'flash fry',
        'roast', 'bake', 'broil', 'grill', 'griddle', 'sear', 'blacken', 'char', 'torch',
        'boil', 'parboil', 'blanch', 'shock', 'simmer', 'poach', 'steam', 'pressure cook',
        'slow cook', 'braise', 'stew', 'confit', 'sous-vide', 'temper', 'toast', 'warm',
        'reheat', 'reduce', 'reduce sauce', 'glaze', 'caramelize', 'candy', 'crystallize',
        'smoke', 'hot smoke', 'cold smoke', 'barbecue', 'rotisserie', 'spit-roast',
        'pan roast', 'oven roast', 'roast whole', 'roast pieces', 'bake blind', 'bake covered',
        'bake uncovered', 'broil high', 'broil low', 'grill marks', 'grill pan', 'plancha',
        'teppanyaki', 'hibachi', 'tandoor', 'clay oven', 'wood-fired', 'coal-fired', 'gas grill',
        'electric grill', 'induction cook', 'microwave', 'solar cook', 'fire pit', 'campfire cook',
        'dutch oven', 'tagine', 'casserole', 'hot pot', 'fondue', 'raclette', 'stone grill'
    ],
    'crunchy': [
        'nuts', 'almonds', 'walnuts', 'pecans', 'hazelnuts', 'peanuts', 'cashews', 'pistachios', 'macadamia nuts',
        'breadcrumbs', 'panko', 'croutons', 'crispy onions', 'fried shallots', 'tempura flakes',
        'seeds', 'pumpkin seeds', 'sunflower seeds', 'sesame seeds', 'flax seeds', 'chia seeds',
        'granola', 'muesli', 'cereal', 'cornflakes', 'rice crispies',
        'raw vegetables', 'celery', 'carrot sticks', 'bell peppers', 'radishes', 'jicama', 'water chestnuts',
        'apple slices', 'pear slices', 'fried items', 'fried chicken skin', 'cracklings', 'pork rinds',
        'chips', 'potato chips', 'tortilla chips', 'plantain chips', 'kale chips', 'parsnip chips',
        'crackers', 'water crackers', 'breadsticks', 'pretzels', 'biscotti',
        'toasted elements', 'toasted coconut', 'candied nuts', 'praline', 'brittle',
        'sugar glass', 'hard candy', 'crispy bacon', 'prosciutto chips', 'crispy sage', 'fried herbs',
        'puffed grains', 'popcorn', 'rice cakes', 'lavash crackers', 'phyllo pastry', 'wonton strips'
    ],
    'creamy': [
        'cream', 'heavy cream', 'whipping cream', 'double cream', 'clotted cream', 'creme fraiche',
        'sour cream', 'mascarpone', 'cream cheese', 'ricotta', 'cottage cheese', 'quark',
        'soft cheeses', 'brie', 'camembert', 'burrata', 'mozzarella', 'fresh cheese',
        'custard', 'creme anglaise', 'pastry cream', 'creme patissiere', 'flan', 'panna cotta',
        'avocado', 'guacamole', 'mashed avocado',
        'purees', 'mashed potatoes', 'sweet potato puree', 'butternut squash puree', 'cauliflower puree',
        'hummus', 'baba ganoush', 'tahini sauce', 'bean dip', 'refried beans',
        'mayonnaise', 'aioli', 'remoulade', 'tartar sauce', 'ranch dressing', 'caesar dressing',
        'creamy soups', 'bisque', 'chowder', 'veloute', 'cream of mushroom', 'cream of tomato',
        'yogurt', 'greek yogurt', 'labneh', 'skyr', 'kefir',
        'coconut milk', 'coconut cream', 'cashew cream', 'almond cream',
        'butter', 'compound butter', 'beurre blanc', 'beurre monte',
        'ganache', 'chocolate mousse', 'cremeux', 'pot de creme',
        'ice cream', 'gelato', 'sorbet', 'frozen yogurt',
        'condensed milk', 'evaporated milk', 'dulce de leche',
        'egg-based sauces', 'hollandaise', 'bearnaise',
        'pureed legumes', 'lentil puree', 'white bean puree', 'black bean dip'
    ],
    'starchy': [
        'potato', 'russet potato', 'yukon gold', 'red potato', 'fingerling', 'sweet potato', 'yam',
        'rice', 'white rice', 'brown rice', 'jasmine rice', 'basmati rice', 'arborio rice', 'sushi rice',
        'pasta', 'spaghetti', 'penne', 'fettuccine', 'macaroni', 'lasagna', 'orzo', 'couscous',
        'bread', 'white bread', 'whole wheat', 'sourdough', 'baguette', 'ciabatta', 'naan', 'pita',
        'flour', 'all-purpose flour', 'bread flour', 'whole wheat flour', 'cornmeal', 'semolina',
        'grains', 'quinoa', 'barley', 'farro', 'freekeh', 'millet', 'oats', 'oatmeal',
        'corn', 'corn kernels', 'polenta', 'grits', 'cornbread', 'tortillas', 'corn tortillas',
        'legumes', 'lentils', 'chickpeas', 'black beans', 'kidney beans', 'pinto beans',
        'root vegetables', 'parsnips', 'turnips', 'rutabaga', 'celeriac', 'taro', 'cassava',
        'winter squash', 'butternut squash', 'acorn squash', 'pumpkin', 'kabocha',
        'processed starches', 'tapioca', 'arrowroot', 'potato starch', 'cornstarch',
        'dumplings', 'gnocchi', 'spaetzle', 'pierogi', 'potato dumplings',
        'breakfast cereals', 'cream of wheat', 'malt-o-meal',
        'plantains', 'green plantains', 'ripe plantains',
        'bread products', 'stuffing', 'bread pudding', 'french toast',
        'pastry', 'pie crust', 'puff pastry', 'shortcrust', 'phyllo dough'
    ]
}

# Separate flavor, texture, and action categories
FLAVOR_CATEGORIES = ['sweet', 'sour', 'umami', 'bitter', 'salty', 'fat', 'spicy']
TEXTURE_CATEGORIES = ['crunchy', 'creamy', 'starchy']
ACTION_CATEGORIES = ['heat_action']


def get_supabase_client() -> Client:
    """Initialize Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials not configured")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def build_category_lookup() -> dict[str, list[str]]:
    """Build reverse lookup: ingredient/action -> list of categories."""
    ingredient_to_categories = defaultdict(list)
    
    for category, items in FILTER_DICT.items():
        for item in items:
            item_lower = item.lower().strip()
            if category not in ingredient_to_categories[item_lower]:
                ingredient_to_categories[item_lower].append(category)
    
    return dict(ingredient_to_categories)


def classify_ingredient(ingredient: str, lookup: dict[str, list[str]]) -> list[str]:
    """
    Classify an ingredient into its flavor/texture categories.
    Uses exact match first, then partial/fuzzy matching.
    """
    ingredient_lower = ingredient.lower().strip()
    
    # Direct match
    if ingredient_lower in lookup:
        return lookup[ingredient_lower]
    
    # Partial match - check if any category item is contained in the ingredient
    categories = []
    for category, items in FILTER_DICT.items():
        if category == 'heat_action':
            continue  # Skip actions for ingredient classification
        for item in items:
            item_lower = item.lower()
            # Check substring match in both directions
            if item_lower in ingredient_lower or ingredient_lower in item_lower:
                if category not in categories:
                    categories.append(category)
                break
    
    return categories


def fetch_all_recipes(supabase: Client) -> list[dict]:
    """Fetch all recipes from the database."""
    print("Fetching recipes from database...")
    
    all_recipes = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("recipes").select(
            "id, name, ingredients, ner, tokens, directions"
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


def compute_recipe_flavor_profile(
    ner: list[str], 
    lookup: dict[str, list[str]]
) -> dict[str, float]:
    """
    Compute a flavor/texture profile for a recipe based on its ingredients.
    Returns a dict mapping category -> intensity score.
    """
    all_categories = FLAVOR_CATEGORIES + TEXTURE_CATEGORIES
    profile = {cat: 0.0 for cat in all_categories}
    
    for ingredient in ner:
        categories = classify_ingredient(ingredient, lookup)
        for cat in categories:
            if cat in profile:
                profile[cat] += 1.0
    
    # Normalize by number of ingredients
    n_ingredients = len(ner) if ner else 1
    for cat in profile:
        profile[cat] /= n_ingredients
    
    return profile


def extract_actions_from_directions(directions: list[str], lookup: dict[str, list[str]]) -> dict[str, float]:
    """
    Extract cooking actions from directions and compute action profile.
    """
    action_profile = {'heat_action': 0.0}
    
    if not directions:
        return action_profile
    
    directions_text = ' '.join(directions).lower()
    
    # Count heat actions found
    heat_action_count = 0
    for action in FILTER_DICT['heat_action']:
        if action.lower() in directions_text:
            heat_action_count += 1
    
    # Normalize by number of direction steps
    n_steps = len(directions) if directions else 1
    action_profile['heat_action'] = min(heat_action_count / n_steps, 1.0)
    
    return action_profile


def build_category_embeddings(
    recipes: list[dict],
    lookup: dict[str, list[str]],
    embedding_dim: int = 64
) -> dict[str, np.ndarray]:
    """
    Learn embeddings for each flavor/texture/action category based on co-occurrence.
    Categories that appear together in recipes will have similar embeddings.
    """
    print("Building category co-occurrence matrix...")
    
    all_categories = FLAVOR_CATEGORIES + TEXTURE_CATEGORIES + ACTION_CATEGORIES
    n_categories = len(all_categories)
    cat_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    
    # Build co-occurrence matrix
    cooccur = np.zeros((n_categories, n_categories), dtype=np.float32)
    
    for recipe in recipes:
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        directions = recipe.get("directions") or []
        if isinstance(directions, str):
            try:
                directions = json.loads(directions)
            except:
                directions = []
        
        # Get categories present in this recipe
        flavor_profile = compute_recipe_flavor_profile(ner, lookup)
        action_profile = extract_actions_from_directions(directions, lookup)
        
        present_categories = []
        for cat, score in flavor_profile.items():
            if score > 0:
                present_categories.append(cat)
        for cat, score in action_profile.items():
            if score > 0:
                present_categories.append(cat)
        
        # Update co-occurrence
        for i, cat1 in enumerate(present_categories):
            for cat2 in present_categories[i:]:
                if cat1 in cat_to_idx and cat2 in cat_to_idx:
                    idx1, idx2 = cat_to_idx[cat1], cat_to_idx[cat2]
                    cooccur[idx1, idx2] += 1
                    if idx1 != idx2:
                        cooccur[idx2, idx1] += 1
    
    # Apply PPMI and SVD
    print("Computing category embeddings via SVD...")
    row_sums = cooccur.sum(axis=1, keepdims=True) + 1e-10
    col_sums = cooccur.sum(axis=0, keepdims=True) + 1e-10
    total = cooccur.sum() + 1e-10
    
    pmi = np.log((cooccur * total) / (row_sums * col_sums) + 1e-10)
    ppmi = np.maximum(pmi, 0)
    
    k = min(embedding_dim, n_categories - 1)
    if k < 1:
        embeddings = np.random.randn(n_categories, embedding_dim).astype(np.float32)
    else:
        sparse_ppmi = csr_matrix(ppmi)
        U, S, Vt = svds(sparse_ppmi, k=k)
        embeddings = U * np.sqrt(S)
        
        if embeddings.shape[1] < embedding_dim:
            padding = np.zeros((n_categories, embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
    
    # Create category -> embedding mapping
    category_embeddings = {}
    for cat, idx in cat_to_idx.items():
        category_embeddings[cat] = embeddings[idx].astype(np.float32)
    
    print(f"  Created embeddings for {len(category_embeddings)} categories")
    return category_embeddings


def build_ingredient_embeddings(
    recipes: list[dict],
    embedding_dim: int = 128
) -> tuple[dict[str, int], np.ndarray]:
    """Build ingredient embeddings using co-occurrence in recipes."""
    print("Building ingredient vocabulary and co-occurrence...")
    
    # Count ingredients
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
    ingredients = [ing for ing, count in ingredient_counts.items() if count >= min_count]
    ing_to_idx = {ing: idx for idx, ing in enumerate(ingredients)}
    n_ingredients = len(ingredients)
    
    print(f"  Vocabulary size: {n_ingredients} ingredients")
    
    if n_ingredients == 0:
        return {}, np.array([])
    
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
        
        for i in range(len(ing_indices)):
            for j in range(i + 1, len(ing_indices)):
                cooccur[ing_indices[i], ing_indices[j]] += 1
                cooccur[ing_indices[j], ing_indices[i]] += 1
    
    # PPMI + SVD
    print("Computing ingredient embeddings via SVD...")
    row_sums = cooccur.sum(axis=1, keepdims=True) + 1e-10
    col_sums = cooccur.sum(axis=0, keepdims=True) + 1e-10
    total = cooccur.sum() + 1e-10
    
    pmi = np.log((cooccur * total) / (row_sums * col_sums) + 1e-10)
    ppmi = np.maximum(pmi, 0)
    
    k = min(embedding_dim, n_ingredients - 1, ppmi.shape[1] - 1)
    if k < 1:
        embeddings = np.random.randn(n_ingredients, embedding_dim).astype(np.float32)
    else:
        sparse_ppmi = csr_matrix(ppmi)
        U, S, Vt = svds(sparse_ppmi, k=k)
        embeddings = U * np.sqrt(S)
        
        if embeddings.shape[1] < embedding_dim:
            padding = np.zeros((n_ingredients, embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
    
    return ing_to_idx, embeddings.astype(np.float32)


def compute_semantic_recipe_embedding(
    recipe: dict,
    ing_to_idx: dict[str, int],
    ing_embeddings: np.ndarray,
    category_embeddings: dict[str, np.ndarray],
    lookup: dict[str, list[str]],
    embedding_dim: int = 512
) -> np.ndarray:
    """
    Compute a semantic embedding for a recipe combining:
    1. Ingredient embeddings (weighted by IDF)
    2. Flavor/texture profile embeddings
    3. Action profile embeddings
    """
    # Parse recipe data
    ner = recipe.get("ner") or []
    if isinstance(ner, str):
        try:
            ner = json.loads(ner)
        except:
            ner = []
    
    directions = recipe.get("directions") or []
    if isinstance(directions, str):
        try:
            directions = json.loads(directions)
        except:
            directions = []
    
    # === Component 1: Ingredient embedding (average) ===
    ing_embed_dim = ing_embeddings.shape[1] if len(ing_embeddings) > 0 else 0
    ing_vec = np.zeros(ing_embed_dim, dtype=np.float32) if ing_embed_dim > 0 else np.zeros(1)
    ing_count = 0
    
    for ing in ner:
        if ing:
            ing_lower = ing.lower().strip()
            if ing_lower in ing_to_idx:
                idx = ing_to_idx[ing_lower]
                ing_vec += ing_embeddings[idx]
                ing_count += 1
    
    if ing_count > 0:
        ing_vec /= ing_count
    
    # === Component 2: Flavor/texture profile embedding ===
    flavor_profile = compute_recipe_flavor_profile(ner, lookup)
    cat_embed_dim = len(list(category_embeddings.values())[0]) if category_embeddings else 0
    flavor_vec = np.zeros(cat_embed_dim, dtype=np.float32) if cat_embed_dim > 0 else np.zeros(1)
    
    for cat, score in flavor_profile.items():
        if cat in category_embeddings and score > 0:
            flavor_vec += score * category_embeddings[cat]
    
    # Normalize flavor vector
    flavor_norm = np.linalg.norm(flavor_vec)
    if flavor_norm > 0:
        flavor_vec /= flavor_norm
    
    # === Component 3: Action profile embedding ===
    action_profile = extract_actions_from_directions(directions, lookup)
    action_vec = np.zeros(cat_embed_dim, dtype=np.float32) if cat_embed_dim > 0 else np.zeros(1)
    
    for cat, score in action_profile.items():
        if cat in category_embeddings and score > 0:
            action_vec += score * category_embeddings[cat]
    
    action_norm = np.linalg.norm(action_vec)
    if action_norm > 0:
        action_vec /= action_norm
    
    # === Combine components with weights ===
    # 50% ingredient, 30% flavor/texture, 20% action
    ing_dim = int(embedding_dim * 0.5)
    flavor_dim = int(embedding_dim * 0.3)
    action_dim = embedding_dim - ing_dim - flavor_dim
    
    def resize_vector(v: np.ndarray, target_dim: int) -> np.ndarray:
        if len(v) == 0:
            return np.zeros(target_dim, dtype=np.float32)
        if len(v) >= target_dim:
            return v[:target_dim]
        else:
            return np.concatenate([v, np.zeros(target_dim - len(v), dtype=np.float32)])
    
    ing_component = resize_vector(ing_vec, ing_dim)
    flavor_component = resize_vector(flavor_vec, flavor_dim)
    action_component = resize_vector(action_vec, action_dim)
    
    full_embedding = np.concatenate([ing_component, flavor_component, action_component])
    
    # L2 normalize
    norm = np.linalg.norm(full_embedding)
    if norm > 0:
        full_embedding /= norm
    
    return full_embedding


def compute_all_recipe_embeddings(
    recipes: list[dict],
    ing_to_idx: dict[str, int],
    ing_embeddings: np.ndarray,
    category_embeddings: dict[str, np.ndarray],
    lookup: dict[str, list[str]],
    embedding_dim: int = 512
) -> dict[int, list[float]]:
    """Compute embeddings for all recipes."""
    print("Computing semantic recipe embeddings...")
    
    recipe_embeddings = {}
    
    for idx, recipe in enumerate(recipes):
        if idx % 500 == 0:
            print(f"  Processing recipe {idx + 1}/{len(recipes)}...")
        
        embedding = compute_semantic_recipe_embedding(
            recipe,
            ing_to_idx,
            ing_embeddings,
            category_embeddings,
            lookup,
            embedding_dim
        )
        
        recipe_embeddings[recipe["id"]] = embedding.tolist()
    
    print(f"  Computed embeddings for {len(recipe_embeddings)} recipes")
    return recipe_embeddings


def update_database_embeddings(
    supabase: Client,
    recipe_embeddings: dict[int, list[float]],
    batch_size: int = 100
) -> None:
    """Update recipe embeddings in the database."""
    print("Updating database with semantic embeddings...")
    
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


def analyze_flavor_distribution(
    recipes: list[dict],
    lookup: dict[str, list[str]]
) -> None:
    """Analyze and print the flavor distribution across recipes."""
    print("\n" + "=" * 60)
    print("FLAVOR & TEXTURE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    category_counts = {cat: 0 for cat in FLAVOR_CATEGORIES + TEXTURE_CATEGORIES}
    
    for recipe in recipes:
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        profile = compute_recipe_flavor_profile(ner, lookup)
        
        for cat in FLAVOR_CATEGORIES + TEXTURE_CATEGORIES:
            if profile.get(cat, 0) > 0:
                category_counts[cat] += 1
    
    print("\nFlavor Categories:")
    for cat in FLAVOR_CATEGORIES:
        count = category_counts[cat]
        pct = count / len(recipes) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:12} {count:5} ({pct:5.1f}%) {bar}")
    
    print("\nTexture Categories:")
    for cat in TEXTURE_CATEGORIES:
        count = category_counts[cat]
        pct = count / len(recipes) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:12} {count:5} ({pct:5.1f}%) {bar}")


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


def verify_embeddings(supabase: Client, lookup: dict, sample_size: int = 5) -> None:
    """Verify embeddings by finding similar recipes and showing flavor profiles."""
    print("\n" + "=" * 60)
    print("VERIFYING EMBEDDINGS WITH SIMILARITY SEARCH")
    print("=" * 60)
    
    response = supabase.table("recipes").select(
        "id, name, ner, embedding"
    ).limit(sample_size).execute()
    
    if not response.data:
        print("  No recipes found for verification")
        return
    
    all_response = supabase.table("recipes").select(
        "id, name, ner, embedding"
    ).limit(100).execute()
    
    all_recipes = all_response.data
    
    for recipe in response.data:
        query_vec = parse_embedding(recipe.get("embedding"))
        if query_vec is None:
            continue
        
        # Get flavor profile
        ner = recipe.get("ner") or []
        if isinstance(ner, str):
            try:
                ner = json.loads(ner)
            except:
                ner = []
        
        profile = compute_recipe_flavor_profile(ner, lookup)
        dominant_flavors = sorted(
            [(k, v) for k, v in profile.items() if v > 0],
            key=lambda x: -x[1]
        )[:3]
        
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
        
        print(f"\n'{recipe['name']}'")
        if dominant_flavors:
            print(f"  Flavor profile: {', '.join([f'{k}({v:.2f})' for k, v in dominant_flavors])}")
        print(f"  Similar recipes:")
        for name, sim in top_3:
            print(f"    - '{name}' (similarity: {sim:.3f})")


def main():
    """Main function to build semantic embeddings."""
    print("=" * 60)
    print("SEMANTIC RECIPE EMBEDDING BUILDER")
    print("Using filter_dict from download_cleaned_data.ipynb")
    print("=" * 60)
    
    # Initialize
    supabase = get_supabase_client()
    lookup = build_category_lookup()
    
    print(f"\nCategory lookup contains {len(lookup)} ingredient/action mappings")
    print(f"Categories: {FLAVOR_CATEGORIES + TEXTURE_CATEGORIES + ACTION_CATEGORIES}")
    
    # Fetch recipes
    recipes = fetch_all_recipes(supabase)
    
    if not recipes:
        print("No recipes found in database!")
        return
    
    # Analyze flavor distribution
    analyze_flavor_distribution(recipes, lookup)
    
    # Build embeddings
    print("\n" + "=" * 60)
    print("BUILDING EMBEDDINGS")
    print("=" * 60)
    
    # 1. Build category embeddings
    category_embeddings = build_category_embeddings(recipes, lookup, embedding_dim=FLAVOR_DIM)
    
    # 2. Build ingredient embeddings
    ing_to_idx, ing_embeddings = build_ingredient_embeddings(recipes, embedding_dim=128)
    
    # 3. Compute recipe embeddings
    recipe_embeddings = compute_all_recipe_embeddings(
        recipes,
        ing_to_idx,
        ing_embeddings,
        category_embeddings,
        lookup,
        EMBEDDING_DIM
    )
    
    # 4. Update database
    update_database_embeddings(supabase, recipe_embeddings)
    
    # 5. Verify
    verify_embeddings(supabase, lookup)
    
    print("\n" + "=" * 60)
    print("SEMANTIC EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
