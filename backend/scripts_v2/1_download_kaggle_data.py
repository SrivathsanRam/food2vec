"""
Filter recipe data from RecipeNLG dataset CSV file.
This script applies strict filtering based on ingredient categories 
to create a high-quality recipe dataset.

Prerequisites:
1. Place RecipeNLG_dataset.csv in the same directory as this script
"""

import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FILENAME = SCRIPT_DIR / '2_strict_filtered_balanced_sample_large_6_ingredients_no_bucket.csv'
RAW_FILENAME = SCRIPT_DIR / 'RecipeNLG_dataset.csv'

# Columns to check for recipe validity
SEARCH_COLS = ['ingredients', 'directions', 'NER']

# Limit rows to process (Set to None for full file)
LIMIT_ROWS = 2231142

# Minimum number of ingredients required
MIN_INGREDIENTS = 6

# Minimum number of steps/directions required
MIN_STEPS = 4

# Maximum recipes to output
MAX_RECIPES = 10000

# Minimum match ratio (percentage of ingredients that must be in allowed list)
MIN_MATCH_RATIO = 1  # 100% of ingredients must match

# Category requirements - recipe must have ingredients from at least N of these categories
REQUIRED_CATEGORY_COUNT = 3  # Must have ingredients from at least 3 different flavor categories

# Flavor categories (exclude actions and textures for balancing)
FLAVOR_CATEGORIES = ['sweet', 'sour', 'umami', 'bitter', 'salty', 'spicy']

# ============================================================================
# STRICT VALIDATION FILTER - Ingredient Categories
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
        'vanilla', 'almond extract', 'coconut milk', 'sweetened yogurt', 'sweet cream', 'marzipan', 'nougat',
        # New Additions
        'simple syrup', 'pears', 'ripe plantains', 'apple butter', 'mango puree', 'lychee', 'pomegranate molasses',
        'sweetened coconut', 'condensed milk', 'evaporated milk', 'sweet tea', 'soda', 'cola'
    ],
    'sour': [
        'vinegar', 'white vinegar', 'apple cider vinegar', 'red wine vinegar', 'balsamic vinegar', 'rice vinegar',
        'champagne vinegar', 'sherry vinegar', 'malt vinegar', 'distilled vinegar', 'lemon', 'lime', 'yogurt',
        'buttermilk', 'sour cream', 'crème fraîche', 'kefir', 'sauerkraut', 'kimchi', 'pickles', 'pickled vegetables',
        'tamarind', 'sumac', 'verjus', 'sorrel', 'rhubarb', 'green apples', 'green grapes', 'gooseberries',
        'cranberries', 'currants', 'passion fruit', 'tart cherries', 'unripe mango', 'green papaya', 'citrus zest',
        'fermented foods', 'kombucha', 'sourdough', 'sour candies', 'citric acid', 'tartaric acid', 'malic acid',
        # New Additions
        'white wine vinegar', 'pickled ginger', 'key lime', 'yuzu', 'sour orange', 'fermented hot sauce', 'sour apple',
        'sour mix', 'sour salt', 'pomegranate seeds', 'green tomato', 'sour plum', 'salad dressing (vinaigrette)'
    ],
    'umami': [
        'soy sauce', 'tamari', 'coconut aminos', 'fish sauce', 'oyster sauce', 'worcestershire sauce', 'hoisin sauce',
        'miso', 'doubanjiang', 'gochujang', 'parmesan cheese', 'pecorino', 'aged gouda', 'blue cheese', 'nutritional yeast',
        'anchovy', 'sardines', 'mackerel', 'bonito flakes', 'dashi', 'kombu', 'nori', 'seaweed', 'mushrooms',
        'shiitake', 'porcini', 'morel', 'chanterelle', 'enoki', 'maitake', 'truffle', 'truffle oil', 'tomato paste',
        'sun-dried tomatoes', 'roasted tomatoes', 'caramelized onions', 'roasted garlic', 'marmite', 'vegemite',
        'bovril', 'beef stock', 'chicken stock', 'vegetable stock', 'bone broth', 'demiglace', 'gravy', 'browned meat',
        'dry-aged beef', 'cured meats', 'fermented beans', 'black garlic', 'soybean paste', 'msg', 'autolyzed yeast extract',
        # New Additions
        'tomato sauce', 'mushroom powder', 'aged balsamic', 'fish paste', 'shrimp paste', 'oyster mushroom', 'king oyster',
        'kelp', 'caviar', 'bottarga', 'dried shrimp', 'beef jerky', 'smoked paprika', 'roasted nuts'
    ],
    'bitter': [
        'cocoa', 'dark chocolate', 'cacao nibs', 'coffee', 'espresso', 'espresso powder', 'matcha', 'green tea',
        'black tea', 'tonic water', 'quinine', 'angostura bitters', 'campari', 'aperol', 'vermouth', 'absinthe',
        'grapefruit', 'bitter melon', 'endive', 'escarole', 'radicchio', 'kale', 'collard greens', 'mustard greens',
        'arugula', 'watercress', 'dandelion greens', 'brussels sprouts', 'broccoli rabe', 'artichokes', 'asparagus',
        'eggplant', 'saffron', 'turmeric', 'fenugreek', 'szechuan peppercorn', 'citrus pith', 'almond skins',
        'walnuts', 'hazelnuts', 'pistachios', 'sesame seeds', 'burnt sugar', 'charred vegetables', 'smoked ingredients',
        'neem', 'gentian root', 'wormwood', 'cascara', 'hops',
        # New Additions
        'unsweetened chocolate', 'cocoa powder', 'escarole', 'frisée', 'chicory', 'dark roast coffee', 'ipa beer',
        'grapefruit peel', 'orange bitters', 'black walnut', 'bitter lettuce', 'burnt onion'
    ],
    'salty': [
        'salt', 'sea salt', 'kosher salt', 'himalayan salt', 'flaky salt', 'smoked salt', 'garlic salt', 'celery salt',
        'soy sauce', 'tamari', 'liquid aminos', 'fish sauce', 'anchovy paste', 'capers', 'olives', 'green olives',
        'kalamata olives', 'castelvetrano olives', 'feta cheese', 'goat cheese', 'halloumi', 'queso fresco', 'cotija',
        'blue cheese', 'gorgonzola', 'pecorino', 'parmesan', 'aged cheddar', 'cured meats', 'bacon', 'pancetta',
        'guanciale', 'prosciutto', 'serrano ham', 'speck', 'salami', 'pepperoni', 'soppressata', 'chorizo', 'sausage',
        'salted butter', 'salted nuts', 'pretzels', 'crackers', 'potato chips', 'popcorn', 'pickles', 'kimchi',
        'sauerkraut', 'fermented vegetables', 'bouillon', 'stock cubes', 'miso paste', 'oyster sauce', 'worcestershire sauce',
        'teriyaki sauce', 'hoisin sauce', 'salted egg yolk', 'salt-packed sardines', 'salt cod', 'biltong', 'jerky',
        # New Additions
        'soy paste', 'cured egg yolk', 'salt pork', 'corned beef', 'pastrami', 'salted caramel', 'salted fish',
        'saltine crackers', 'tortilla chips', 'salted pretzels', 'pickle juice', 'brine', 'seaweed salad'
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
        'caesar dressing', 'tahini', 'chocolate', 'cocoa butter',
        # New Additions
        'pork fat', 'lamb fat', 'bone marrow', 'fatback', 'suet', 'clarified drippings', 'sesame paste',
        'sunflower seed butter', 'coconut cream', 'whipped cream', 'clotted cream', 'double cream', 'infused oils'
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
        'cajun seasoning', 'creole seasoning', 'berbere', 'ras el hanout', 'curry powder', 'curry paste',
        # New Additions
        'aleppo pepper', 'gochugaru', 'shichimi togarashi', 'old bay seasoning', 'blackening seasoning',
        'chili oil', 'chili crisp', 'fresh horseradish', 'prepared horseradish', 'wasabi paste', 'ginger paste'
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
        'dutch oven', 'tagine', 'casserole', 'hot pot', 'fondue', 'raclette', 'stone grill',
        # New Additions
        'sear off', 'brown', 'fry off', 'crisp', 'crisp up', 'render', 'melt', 'scald', 'scorch',
        'char-broil', 'flame-grill', 'griddle cook', 'hot plate', 'salt bake', 'clay pot cook'
    ],
    'mech_action': [
        'chop', 'dice', 'fine dice', 'brunoise', 'mince', 'julienne', 'batons', 'chiffonade',
        'slice', 'bias cut', 'roll cut', 'cube', 'wedge', 'crush', 'press', 'grate', 'shred',
        'zest', 'peel', 'segment', 'supreme', 'fillet', 'bone', 'skin', 'trim', 'clean',
        'mix', 'stir', 'fold', 'combine', 'toss', 'turn', 'coat', 'dredge', 'bread',
        'batter', 'dip', 'blend', 'puree', 'mash', 'smash', 'cream', 'whip', 'beat',
        'whisk', 'emulsify', 'sift', 'strain', 'drain', 'rinse', 'soak', 'hydrate',
        'marinate', 'brine', 'dry brine', 'wet brine', 'cure', 'rest', 'rise', 'proof',
        'knead', 'roll', 'shape', 'form', 'portion', 'scale', 'measure', 'weigh',
        'scoop', 'spoon', 'ladle', 'pour', 'drizzle', 'splash', 'dash', 'pinch',
        'season', 'salt', 'pepper', 'spice', 'herb', 'garnish', 'plate', 'arrange',
        'decorate', 'pipe', 'spread', 'layer', 'stack', 'assemble', 'build', 'construct',
        'wrap', 'roll up', 'tie', 'truss', 'skewer', 'thread', 'stuff', 'fill',
        'inject', 'brush', 'baste', 'glaze', 'polish', 'shine', 'cool', 'chill',
        'freeze', 'thaw', 'defrost', 'temper', 'set', 'gel', 'jellify', 'congeal',
        'clarify', 'filter', 'decant', 'settle', 'separate', 'clarify butter', 'render fat',
        # New Additions
        'cut', 'rough chop', 'fine chop', 'cut into', 'break down', 'portion out', 'tear', 'pull apart',
        'crush garlic', 'pound', 'tenderize', 'score', 'notch', 'perforate', 'core', 'pit', 'devein',
        'whisk together', 'stir in', 'fold in', 'beat until', 'cream together', 'cut in', 'rub in'
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
        'puffed grains', 'popcorn', 'rice cakes', 'lavash crackers', 'phyllo pastry', 'wonton strips',
        # New Additions
        'fried garlic', 'fried leeks', 'crispy shallots', 'crispy chickpeas', 'soy nuts', 'wasabi peas',
        'crispy rice', 'crackling', 'pork crackling', 'chicharron', 'crispy wontons', 'spring roll wrapper',
        'crispy noodles', 'chow mein noodles', 'fried rice noodles'
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
        'pureed legumes', 'lentil puree', 'white bean puree', 'black bean dip',
        # New Additions
        'alfredo sauce', 'bechamel sauce', 'cheese sauce', 'fondue', 'whipped feta',
        'cream of chicken soup', 'vichyssoise', 'potato leek soup', 'cream of asparagus',
        'clotted cream', 'devonshire cream', 'crema', 'crema mexicana', 'sour cream dip'
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
        'pastry', 'pie crust', 'puff pastry', 'shortcrust', 'phyllo dough',
        # New Additions
        'hash browns', 'tater tots', 'potato pancakes', 'latkes', 'fried rice', 'risotto', 'paella',
        'udon noodles', 'soba noodles', 'ramen noodles', 'rice noodles', 'vermicelli', 'angel hair pasta',
        'egg noodles', 'spätzle', 'matzo', 'flatbread', 'focaccia', 'rye bread', 'pumpernickel',
        'corn dogs', 'hush puppies', 'polenta fries', 'potato salad', 'pasta salad'
    ],
    'protein': [
        'egg', 'eggs', 'chicken', 'beef', 'pork', 'turkey', 'lamb', 'fish', 'shrimp',
        'salmon', 'tuna', 'tofu', 'tempeh', 'ground beef', 'chicken breast', 'bacon',
        'ham', 'sausage', 'ground turkey', 'crab', 'lobster', 'scallops', 'mussels',
        # Major Expansions: Meats (Cuts, Forms, States)
        'chicken thighs', 'chicken legs', 'chicken wings', 'chicken drumsticks', 'whole chicken', 'rotisserie chicken', 'chicken tenderloins',
        'chicken strips', 'chicken nuggets', 'chicken sausage', 'ground chicken', 'chicken liver', 'duck', 'duck breast', 'duck confit',
        'goose', 'quail', 'cornish hen', 'rabbit', 'venison', 'bison', 'elk', 'wild boar',
        'beef steak', 'beef roast', 'beef chuck', 'beef brisket', 'beef short ribs', 'beef ribs', 'beef shank', 'beef tenderloin',
        'filet mignon', 'ribeye', 'sirloin', 'flank steak', 'skirt steak', 'hanger steak', 'beef tri-tip', 'beef tongue',
        'beef liver', 'ground pork', 'pork chop', 'pork loin', 'pork tenderloin', 'pork shoulder', 'pork belly', 'pork ribs',
        'baby back ribs', 'spare ribs', 'pork sausage', 'italian sausage', 'chorizo', 'kielbasa', 'bratwurst', 'andouille',
        'lamb chops', 'lamb leg', 'lamb shoulder', 'lamb shank', 'ground lamb', 'goat', 'goat meat',
        # Fish & Seafood (Varieties, Cuts, Forms)
        'white fish', 'cod', 'haddock', 'halibut', 'sole', 'flounder', 'tilapia', 'catfish', 'trout',
        'mackerel', 'sardines', 'anchovies', 'herring', 'swordfish', 'mahi mahi', 'sea bass', 'branzino',
        'barramundi', 'monkfish', 'pollock', 'orange roughy', 'arctic char', 'sturgeon', 'eel', 'octopus',
        'squid', 'calamari', 'crawfish', 'crayfish', 'clams', 'oysters', 'abalone', 'sea urchin',
        'fish fillets', 'fish steaks', 'whole fish', 'smoked fish', 'canned fish', 'fish cakes', 'surimi',
        'imitation crab', 'breaded fish', 'fish sticks',
        # Plant-Based & Specialty Proteins
        'seitan', 'textured vegetable protein', 'tvp', 'soy curls', 'jackfruit', 'mycoprotein', 'quorn',
        'lupini beans', 'edamame', 'soybeans', 'soy milk', 'soy yogurt', 'pea protein', 'hemp seeds',
        'spirulina', 'nutritional yeast', 'chickpeas', 'black beans', 'kidney beans', 'lentils', 'pinto beans',
        # Processed & Prepared Proteins
        'deli meat', 'sliced turkey', 'sliced ham', 'roast beef', 'pastrami', 'corned beef', 'salami',
        'pepperoni', 'prosciutto', 'pancetta', 'speck', 'biltong', 'jerky', 'beef jerky', 'turkey jerky',
        'meatballs', 'meatloaf', 'patty', 'burger patty', 'veggie burger', 'black bean burger',
        'hot dog', 'frankfurter', 'weenie', 'sausage links', 'sausage patties',
        # Eggs & Egg Products
        'egg whites', 'egg yolks', 'liquid eggs', 'powdered eggs', 'hard-boiled eggs', 'soft-boiled eggs',
        'poached eggs', 'scrambled eggs', 'fried eggs', 'omelette', 'frittata', 'quiche', 'deviled eggs',
        'egg salad', 'century egg', 'duck egg', 'quail egg', 'goose egg'
    ],

    'dairy': [
        'milk', 'whole milk', 'skim milk', 'cheese', 'cheddar', 'mozzarella',
        'parmesan', 'cream', 'butter', 'eggs', 'egg whites', 'egg yolks',
        # Milk & Milk Variants (Types, Fat Contents, Forms)
        '1% milk', '2% milk', 'reduced fat milk', 'fat-free milk', 'lactose-free milk', 'raw milk',
        'pasteurized milk', 'homogenized milk', 'evaporated milk', 'condensed milk', 'sweetened condensed milk',
        'powdered milk', 'dry milk', 'goat milk', 'sheep milk', 'buffalo milk', 'camel milk',
        'buttermilk', 'kefir', 'acidophilus milk', 'flavored milk', 'chocolate milk', 'strawberry milk',
        # Cream & Cream Products
        'light cream', 'table cream', 'whipping cream', 'heavy whipping cream', 'double cream', 'clotted cream',
        'sour cream', 'creme fraiche', 'cultured cream', 'cream top', 'cream layer',
        # Yogurt & Cultured Dairy
        'plain yogurt', 'vanilla yogurt', 'greek yogurt', 'icelandic yogurt', 'skyr', 'australian yogurt',
        'bulgarian yogurt', 'indian yogurt', 'dahi', 'labneh', 'yogurt cheese', 'drinkable yogurt',
        'frozen yogurt', 'yogurt powder',
        # Cheese (Major Categories and Varieties)
        'soft cheese', 'semi-soft cheese', 'semi-hard cheese', 'hard cheese', 'fresh cheese',
        'blue cheese', 'washed-rind cheese', 'bloomy-rind cheese', 'pressed cheese',
        'american cheese', 'colby', 'colby-jack', 'monterey jack', 'pepper jack',
        'swiss cheese', 'emmental', 'gruyere', 'raclette', 'fontina',
        'provolone', 'gouda', 'edam', 'havarti', 'muenster',
        'feta', 'goat cheese', 'chevre', 'brie', 'camembert',
        'ricotta', 'cottage cheese', 'paneer', 'queso fresco', 'cotija',
        'halloumi', 'manchego', 'asiago', 'pecorino', 'romano',
        # Butter & Butter Products
        'salted butter', 'unsalted butter', 'cultured butter', 'european-style butter',
        'clarified butter', 'ghee', 'brown butter', 'beurre noisette', 'compound butter',
        'whipped butter', 'spreadable butter', 'butter blends', 'margarine',
        # Dairy Alternatives (Plant-Based)
        'almond milk', 'soy milk', 'oat milk', 'coconut milk', 'rice milk', 'hemp milk',
        'cashew milk', 'pea milk', 'flax milk', 'plant milk', 'non-dairy milk',
        'vegan cheese', 'nut cheese', 'soy cheese', 'coconut yogurt', 'almond yogurt',
        'vegan butter', 'plant-based butter', 'coconut cream', 'canned coconut milk'
    ],

    'basics': [
        'water', 'flour', 'all-purpose flour', 'baking powder', 'baking soda',
        'yeast', 'cornstarch', 'bread crumbs', 'oats', 'sugar', 'salt', 'pepper',
        # Flours & Meals
        'self-rising flour', 'cake flour', 'pastry flour', 'bread flour', 'whole wheat flour',
        'white whole wheat flour', 'spelt flour', 'rye flour', 'buckwheat flour',
        'almond flour', 'coconut flour', 'oat flour', 'rice flour', 'gluten-free flour blend',
        'cornmeal', 'polenta', 'semolina', 'farina', 'masa harina',
        # Leaveners & Thickeners
        'active dry yeast', 'instant yeast', 'fresh yeast', 'sourdough starter',
        'cream of tartar', 'baking powder', 'double-acting baking powder',
        'baking soda', 'bicarbonate of soda',
        'arrowroot powder', 'tapioca starch', 'potato starch', 'xanthan gum', 'guar gum',
        'psyllium husk', 'gelatin', 'agar agar', 'pectin',
        # Sugars & Sweet Granules
        'granulated sugar', 'white sugar', 'caster sugar', 'superfine sugar',
        'brown sugar', 'dark brown sugar', 'light brown sugar', 'demerara sugar',
        'turbinado sugar', 'raw sugar', 'muscovado sugar', 'coconut sugar',
        'powdered sugar', 'confectioners sugar', 'icing sugar',
        'sanding sugar', 'pearl sugar', 'rock sugar', 'candy sugar',
        # Salts
        'table salt', 'iodized salt', 'kosher salt', 'coarse salt', 'fine sea salt',
        'flake salt', 'sel gris', 'fleur de sel', 'himalayan pink salt', 'black salt',
        'smoked salt', 'garlic salt', 'onion salt', 'celery salt', 'seasoned salt',
        # Peppers & Basic Spices
        'black peppercorns', 'ground black pepper', 'white pepper', 'green peppercorns',
        'pink peppercorns', 'szechuan peppercorns', 'tellicherry pepper',
        'cayenne pepper', 'crushed red pepper', 'red pepper flakes',
        'paprika', 'smoked paprika', 'sweet paprika', 'hot paprika',
        'garlic powder', 'onion powder', 'ground cumin', 'cumin seeds',
        'chili powder', 'curry powder', 'italian seasoning', 'herbes de provence',
        # Oils & Cooking Fats
        'vegetable oil', 'canola oil', 'rapeseed oil', 'sunflower oil', 'safflower oil',
        'corn oil', 'soybean oil', 'peanut oil', 'grapeseed oil',
        'olive oil', 'extra virgin olive oil', 'light olive oil', 'pomace oil',
        'coconut oil', 'avocado oil', 'sesame oil', 'toasted sesame oil',
        'walnut oil', 'almond oil', 'hazelnut oil', 'pumpkin seed oil',
        'lard', 'tallow', 'duck fat', 'schmaltz', 'shortening',
        # Vinegars & Acidic Basics
        'white vinegar', 'distilled vinegar', 'apple cider vinegar',
        'red wine vinegar', 'white wine vinegar', 'champagne vinegar',
        'balsamic vinegar', 'aged balsamic', 'balsamic glaze',
        'rice vinegar', 'seasoned rice vinegar', 'black vinegar',
        'malt vinegar', 'sherry vinegar', 'persimmon vinegar',
        'lemon juice', 'lime juice', 'citric acid', 'sumac',
        # Liquid Bases
        'water', 'filtered water', 'mineral water', 'sparkling water',
        'club soda', 'soda water', 'tonic water',
        'stock', 'broth', 'bouillon', 'bouillon cube', 'bouillon powder',
        'beef stock', 'chicken stock', 'vegetable stock', 'fish stock',
        'bone broth', 'dashi', 'kombu dashi', 'bonito dashi',
        # Pan Coatings & Binders
        'cooking spray', 'parchment paper', 'wax paper', 'aluminum foil',
        'butter wrapper', 'oil mister',
        'breadcrumbs', 'panko breadcrumbs', 'italian breadcrumbs',
        'crushed crackers', 'graham cracker crumbs', 'cookie crumbs'
    ]
}


def build_allowed_ingredients() -> set:
    """Build the set of allowed ingredients from filter dict."""
    allowed = set()
    for category, items in FILTER_DICT.items():
        for item in items:
            allowed.add(item.lower())
    return allowed


def build_category_lookup() -> dict[str, set]:
    """Build a lookup from ingredient to its categories."""
    ingredient_to_categories = {}
    for category, items in FILTER_DICT.items():
        for item in items:
            item_lower = item.lower()
            if item_lower not in ingredient_to_categories:
                ingredient_to_categories[item_lower] = set()
            ingredient_to_categories[item_lower].add(category)
    return ingredient_to_categories


def get_recipe_categories(ner_list: list, category_lookup: dict) -> set:
    """Get all categories represented in a recipe."""
    categories = set()
    for ingredient in ner_list:
        ing_lower = ingredient.lower()
        if ing_lower in category_lookup:
            categories.update(category_lookup[ing_lower])
    return categories


def validate_recipe(ner_list: list, directions: list, allowed_ingredients: set, category_lookup: dict) -> tuple[bool, float]:
    """
    Validate a recipe based on:
    1. Has at least MIN_INGREDIENTS ingredients
    2. Has at least MIN_STEPS directions
    3. At least MIN_MATCH_RATIO of ingredients are in the allowed list
    4. Has ingredients from at least REQUIRED_CATEGORY_COUNT flavor categories
    
    Returns:
        (is_valid, match_ratio)
    """
    if len(ner_list) < MIN_INGREDIENTS:
        return False, 0.0
    
    if len(directions) < MIN_STEPS:
        return False, 0.0
    
    # Calculate match ratio
    matched = sum(1 for ing in ner_list if ing.lower() in allowed_ingredients)
    match_ratio = matched / len(ner_list)
    
    if match_ratio < MIN_MATCH_RATIO:
        return False, match_ratio
    
    # Check flavor category diversity
    recipe_categories = get_recipe_categories(ner_list, category_lookup)
    flavor_categories_present = recipe_categories.intersection(set(FLAVOR_CATEGORIES))
    
    if len(flavor_categories_present) < REQUIRED_CATEGORY_COUNT:
        return False, match_ratio
    
    return True, match_ratio


def filter_recipes(input_path: Path, output_path: Path, limit_rows: int = None) -> int:
    """
    Filter recipes from the raw Kaggle CSV based on ingredient validation.
    Limits output to MAX_RECIPES with balanced category distribution.
    
    Returns:
        Number of valid recipes found
    """
    print(f"\n🔍 Filtering recipes from: {input_path.name}")
    print(f"   Minimum ingredients: {MIN_INGREDIENTS}")
    print(f"   Minimum steps: {MIN_STEPS}")
    print(f"   Minimum match ratio: {MIN_MATCH_RATIO * 100}%")
    print(f"   Required flavor categories: {REQUIRED_CATEGORY_COUNT}")
    print(f"   Max recipes: {MAX_RECIPES}")
    
    allowed_ingredients = build_allowed_ingredients()
    category_lookup = build_category_lookup()
    print(f"   Allowed ingredients in filter: {len(allowed_ingredients)}")
    
    all_valid_recipes = []
    category_counts = {cat: 0 for cat in FLAVOR_CATEGORIES}
    
    chunk_size = 50000
    total_rows = limit_rows or LIMIT_ROWS
    
    with tqdm(total=total_rows, unit='rows', desc="Scanning") as pbar:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, on_bad_lines='skip', nrows=total_rows):
            for idx, row in chunk.iterrows():
                # Stop if we have enough recipes
                if len(all_valid_recipes) >= MAX_RECIPES:
                    break
                    
                try:
                    ner_list = ast.literal_eval(row['NER'])
                    directions = ast.literal_eval(row['directions'])
                    is_valid, match_ratio = validate_recipe(ner_list, directions, allowed_ingredients, category_lookup)
                    
                    if is_valid:
                        # Get primary category for balancing
                        recipe_cats = get_recipe_categories(ner_list, category_lookup)
                        flavor_cats = recipe_cats.intersection(set(FLAVOR_CATEGORIES))
                        
                        # Find the least represented category this recipe belongs to
                        if flavor_cats:
                            min_cat = min(flavor_cats, key=lambda c: category_counts.get(c, 0))
                            category_counts[min_cat] += 1
                        
                        all_valid_recipes.append(row)
                        
                except (ValueError, SyntaxError):
                    continue
            
            pbar.update(len(chunk))
            
            if len(all_valid_recipes) >= MAX_RECIPES:
                print(f"\n   Reached {MAX_RECIPES} recipes, stopping...")
                break
    
    # Create final DataFrame and remove duplicates
    print("\n📊 Consolidating results...")
    final_df = pd.DataFrame(all_valid_recipes).drop_duplicates(subset=['title', 'link'])
    
    # If still too many, sample down
    if len(final_df) > MAX_RECIPES:
        final_df = final_df.sample(n=MAX_RECIPES, random_state=42)
    
    # Save to output file
    final_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(final_df)} unique valid recipes to: {output_path.name}")
    
    # Print category distribution
    print("\n📊 Category distribution in final dataset:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")
    
    return len(final_df)


def print_dataset_stats(csv_path: Path):
    """Print statistics about the filtered dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    df = pd.read_csv(csv_path, nrows=1000)
    
    print(f"\nColumns: {list(df.columns)}")
    print(f"Sample size: {len(df)} (first 1000 rows)")
    
    # Count ingredients per recipe
    ing_counts = []
    for ner_str in df['NER'].dropna():
        try:
            ner_list = ast.literal_eval(ner_str)
            ing_counts.append(len(ner_list))
        except:
            pass
    
    if ing_counts:
        print(f"\nIngredients per recipe:")
        print(f"  Min: {min(ing_counts)}")
        print(f"  Max: {max(ing_counts)}")
        print(f"  Avg: {sum(ing_counts) / len(ing_counts):.1f}")
    
    # Sample recipes
    print(f"\nSample recipe titles:")
    for title in df['title'].head(5):
        print(f"  - {title}")


def main():
    """Main function to filter recipe data from CSV."""
    print("=" * 60)
    print("RECIPE DATA FILTER")
    print("=" * 60)
    
    # Check if filtered data already exists
    if OUTPUT_FILENAME.exists():
        print(f"\n📁 Filtered data already exists: {OUTPUT_FILENAME.name}")
        print_dataset_stats(OUTPUT_FILENAME)
        
        response = input("\nRe-filter the data? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing filtered data.")
            return
    
    # Check for raw CSV file
    if not RAW_FILENAME.exists():
        print(f"\n❌ CSV file not found: {RAW_FILENAME}")
        print("Please place RecipeNLG_dataset.csv in the scripts_v2 directory.")
        return
    
    print(f"\n📁 Using CSV file: {RAW_FILENAME.name}")
    
    # Filter the data
    num_recipes = filter_recipes(RAW_FILENAME, OUTPUT_FILENAME, LIMIT_ROWS)
    
    if num_recipes > 0:
        print_dataset_stats(OUTPUT_FILENAME)
        print("\n" + "=" * 60)
        print("FILTERING COMPLETE!")
        print(f"Output: {OUTPUT_FILENAME}")
        print("=" * 60)
    else:
        print("\n❌ No valid recipes found after filtering.")


if __name__ == "__main__":
    main()
