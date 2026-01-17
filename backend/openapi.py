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
    response = openai_query("I need you to return a json which contains nodes, where each node contains a increasing numeric id, a type (ingredient, intermediate or final), label which contains the ingredient name for ingredients or Step (number) result for intermediate or final steps, and state which is either raw for ingredients or the action perform at that step. The json should also contain edges where it consists of steps, action, source and the target, where the source and target is the node id. The json should also contain metadata which contains step_count, total_edges, total_nodes, ingredient_count.", 
                            steps)
    text_content = response.output[0].content[0].text

    # Remove markdown code blocks
    clean_text = re.sub(r'```python\n?', '', text_content)
    clean_text = re.sub(r'```\n?', '', clean_text).lstrip('json')
    
    # Convert from string to json format
    json_obj = json.loads(clean_text)
    
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