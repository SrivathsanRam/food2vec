import re
import ast
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)

def openai_query(instructions, query_input):
    """Send query to openai to get a response"""
    response = client.responses.create(
        model="gpt-4o",
        instructions=instructions,
        input=query_input,
    )
    
    return response

def extract_tokens(steps):
    response = openai_query("I need you to return an array of tokens where each individual token is represented by either TEMP_(temperature) or ACT_(action) or ING_(ingredient). There should be an underscore between words if there are mode than 2 words. The temperature should just be the numerical value and scale (where the scale is of fahrenheit, kelvin or celsius) and not include any timings. The action should consists of a single word and it should be in lowercase. The ingredients should not include adjectives or any measurements for how big it is or any measurement items like sheets or any equipment. Make sure you preserve the original order strictly by sentence", 
                            steps)
    text_content = response.output[0].content[0].text
    
    # Remove markdown code blocks
    clean_text = re.sub(r'```python\n?', '', text_content)
    clean_text = re.sub(r'```\n?', '', clean_text)
    
    # Convert from string to array form    
    tokens_array = re.findall(r'"([^"]*)"', clean_text)
        
    return tokens_array

def extract_graph_representation(steps):
    response = openai_query(
        """I need you to return a JSON which contains nodes and edges for a recipe graph.

IMPORTANT: Node IDs must be 0-indexed (start from 0, not 1) and be sequential (0, 1, 2, 3...).

Format:
- "nodes": array of {"id": int (0-indexed), "type": "ingredient"|"intermediate"|"final", "label": string, "state": string}
  - Ingredients have type "ingredient", label is the ingredient name, state is "raw"
  - Intermediate steps have type "intermediate", label is "Step N result", state is the action performed
  - Final dish has type "final", label is "Final dish", state is the final action

- "edges": array of {"step": int, "action": string, "source": int, "target": int}
  - source and target must be valid node IDs (0-indexed, matching nodes array)
  - source is the input node, target is the output node

- "metadata": {"step_count": int, "total_edges": int, "total_nodes": int, "ingredient_count": int}

Example for 3 ingredients combined in 2 steps:
{
  "nodes": [
    {"id": 0, "type": "ingredient", "label": "flour", "state": "raw"},
    {"id": 1, "type": "ingredient", "label": "sugar", "state": "raw"},
    {"id": 2, "type": "ingredient", "label": "butter", "state": "raw"},
    {"id": 3, "type": "intermediate", "label": "Step 1 result", "state": "mixed"},
    {"id": 4, "type": "final", "label": "Final dish", "state": "baked"}
  ],
  "edges": [
    {"step": 1, "action": "mix", "source": 0, "target": 3},
    {"step": 1, "action": "mix", "source": 1, "target": 3},
    {"step": 1, "action": "mix", "source": 2, "target": 3},
    {"step": 2, "action": "bake", "source": 3, "target": 4}
  ],
  "metadata": {"step_count": 2, "total_edges": 4, "total_nodes": 5, "ingredient_count": 3}
}""",
        steps
    )
    text_content = response.output[0].content[0].text

    # Remove markdown code blocks
    clean_text = re.sub(r'```python\n?', '', text_content)
    clean_text = re.sub(r'```\n?', '', clean_text).lstrip('json')
    
    # Convert from string to json format
    try:
        json_obj = json.loads(clean_text)
    except json.JSONDecodeError:
        # Return empty graph if parsing fails
        json_obj = {"nodes": [], "edges": [], "metadata": {}}
    
    return json_obj
    

def extract_directions(steps):
    response = openai_query("Extract out all of the steps and store it in an array in python format", steps)
    text_content = response.output[0].content[0].text
    
    # Remove markdown code blocks
    clean_text = re.sub(r'```python\n?', '', text_content)
    clean_text = re.sub(r'```\n?', '', clean_text)
    
    # Convert from string to array form    
    steps_array = re.findall(r'"([^"]*)"', clean_text)
    
    return steps_array


def extract_ingredients(steps):
    response = openai_query("Extract out all of the ingredients in the steps listed and return the result in an array in python format", steps)
    text_content = response.output[0].content[0].text
    
    # Remove markdown code blocks
    clean_text = re.sub(r'```python\n?', '', text_content)
    clean_text = re.sub(r'```\n?', '', clean_text)
    
    # Extract the list from "ingredients = [...]"
    match = re.search(r'ingredients\s*=\s*(\[.*?\])', clean_text, re.DOTALL)
    if match:
        list_str = match.group(1)
        ingredients = ast.literal_eval(list_str)
        return ingredients
    
    return []


def generate_recipe_steps(recipe_name):
    """Generate recipe steps from a recipe name using AI."""
    response = openai_query(
        "You are a professional chef. Given a recipe name, generate detailed cooking instructions. "
        "Return ONLY the step-by-step cooking instructions as a single paragraph. "
        "Include specific ingredients with measurements and detailed cooking techniques. "
        "Make it realistic and delicious.",
        f"Generate detailed cooking steps for: {recipe_name}"
    )
    text_content = response.output[0].content[0].text
    return text_content.strip()