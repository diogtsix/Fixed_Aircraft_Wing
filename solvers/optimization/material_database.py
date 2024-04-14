import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataobjects.material import Material
import numpy as np
"""

# Load the material database
with open('Utilities/material_database.json', 'r') as file:
    material_db = json.load(file)

# Function to create a Material instance based on material name
def get_material(material_name):
    if material_name in material_db:
        properties = material_db[material_name]
        return Material(**properties)  # Unpack properties into Material constructor
    else:
        raise ValueError(f"Material '{material_name}' not found in database.")
    
# Example usage
aluminum = get_material("Titanium")
print(aluminum)
"""

def load_material_database(json_filepath='Utilities/material_database.json'):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    absolute_path = os.path.join(project_root, json_filepath)

    with open(absolute_path, 'r') as file:
        material_db = json.load(file)
    return material_db

def generate_material_matrix(json_filepath='Utilities/material_database.json'):
    material_db = load_material_database(json_filepath)
    material_matrix = []
    
    for material_name, properties in material_db.items():
        #material_id = properties.pop('id', None) 
        material_obj = Material(**properties)
        material_obj.name = material_name
        #material_matrix.append([material_name, material_id, material_obj])
        material_matrix.append([material_name, material_obj.id, material_obj])

    
    return material_matrix

# If you prefer a NumPy array instead of a list
def generate_material_np_matrix(json_filepath='Utilities/material_database.json'):
    material_matrix = generate_material_matrix(json_filepath)
    material_np_matrix = np.array(material_matrix, dtype=object)  # Use dtype=object for mixed types
    return material_np_matrix

