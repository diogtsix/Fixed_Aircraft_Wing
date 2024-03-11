import numpy as np
from dataobjects.naca_airfoil_generation import naca2D
from dataobjects.material import Material
from dataobjects.node import Node

class Preprocessor():
    
    def __init__(self, chordLength = 1.12, wing_length = 3.61, 
                 elementMaterial = Material, elementSurface = 0.0031):
        
        self.chordLength = chordLength
        self.wing_length = wing_length
        self.elementMaterial = elementMaterial
        self.elementSurface = elementSurface

        
        self.elementMaterial = Material()
        
        nodes_airfoil, elements_airfoil = naca2D(chordLength, plotAirfoil = False)
        
        self.nodesAirfoil = nodes_airfoil
        self.elementsAirfoil = elements_airfoil
          
        self.nodeMatrix = self.createWingNodeMatrix()
        self.elementMatrix = self.createAirfoilElementMatrix()
        
    
    
    def createWingNodeMatrix(self):
        
        airfoilNodeMatrix = self.createAirfoilNodeMatrix()
        
        
        return airfoilNodeMatrix
    
    def createAirfoilNodeMatrix(self):
        
        
        # Initialize an empty list to store the Node objects
        node_objects = []
        dof_start = 1

        # Loop through each row in the nodes_airfoil matrix
        for row in self.nodesAirfoil:
            x, y = row[:2]  # Extract X and Y coordinates
            coords = np.array([x, y, 0])  # Set Z coordinate to 0
            
            # Calculate the current dof_id array for this node
            current_dof_id = np.arange(dof_start, dof_start + 6)
    
    
            # Instantiate a Node object with the specified coordinates and default values for other parameters
            node = Node(coords=coords , dof_id=current_dof_id)
            # Add the newly created Node object to the list
            node_objects.append(node)
            
            # Update dof_start for the next node
            dof_start += 6
            
        return node_objects
    
    def createAirfoilElementMatrix(self):
        pass