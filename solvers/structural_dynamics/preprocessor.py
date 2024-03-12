import numpy as np
from dataobjects.naca_airfoil_generation import naca2D
from dataobjects.material import Material
from dataobjects.node import Node

class Preprocessor():
    
    def __init__(self, chordLength = 1.12, wing_length = 3.61, 
                 elementMaterial = Material, elementSurface = 0.0031, 
                 numberOfAirfoils = 11):
        
        self.chordLength = chordLength
        self.wing_length = wing_length
        self.elementMaterial = elementMaterial
        self.elementSurface = elementSurface
        self.numberOfAirfoils = numberOfAirfoils

        
        self.elementMaterial = Material()
        
        nodes_airfoil, elements_airfoil = naca2D(chordLength, plotAirfoil = False)
        
        self.nodesAirfoil = nodes_airfoil
        self.elementsAirfoil = elements_airfoil
          
        self.nodeMatrix = self.createWingNodeMatrix()
        self.elementMatrix = self.createWingElementMatrix()
        
    
    
    def createWingNodeMatrix(self):
        
        airfoilNodeMatrix = self.createAirfoilNodeMatrix()
        
        
        if self.numberOfAirfoils > 1 :
            d = self.wing_length/(self.numberOfAirfoils - 1)
        else :
            d = 0
            
        counterNodes = 0
        counterDofs = 1
        
        # Initialize an empty list to store the wing nodes
        nodesWing = []
        nodeID = 1

            
        # Generate wing nodes
        for i in range(self.numberOfAirfoils):
            for j in range(12):  # 12 nodes per airfoil
                node = airfoilNodeMatrix[j]
                new_coords = np.array([node.coords[0], node.coords[1], (i * d)])  # Update Z coordinate
                new_dof_id = np.arange(counterDofs, counterDofs + 6)  # Update dof_ids
        
                # Create a new Node object for the wing node
                new_node = Node(coords=new_coords, dof_id=new_dof_id, node_id = nodeID)
                nodesWing.append(new_node)
        
                # Update counters
                counterDofs += 6
                nodeID += 1
            counterNodes += 12
        return nodesWing
    
    def createAirfoilNodeMatrix(self):
        
        
        # Initialize an empty list to store the Node objects
        node_objects = []
        dof_start = 1
        node_id = 1

        # Loop through each row in the nodes_airfoil matrix
        for row in self.nodesAirfoil:
            x, y = row[:2]  # Extract X and Y coordinates
            coords = np.array([x, y, 0])  # Set Z coordinate to 0
            
            # Calculate the current dof_id array for this node
            current_dof_id = np.arange(dof_start, dof_start + 6)
    
    
            # Instantiate a Node object with the specified coordinates and default values for other parameters
            node = Node(coords=coords , dof_id=current_dof_id, node_id = node_id)
            # Add the newly created Node object to the list
            node_objects.append(node)
            
            # Update dof_start for the next node
            dof_start += 6
            node_id += 1
            
        return node_objects
    
    def createAirfoilElementMatrix(self):
        
        # Initialize an empty list for airfoilElementMatrix
        airfoilElementMatrix = []
        for element in self.elementsAirfoil:
            node_id_1, node_id_3 = element[0], element[1]
            
            # Find the corresponding Node objects
            node_1 = self.find_node_by_id(node_id_1, self.nodeMatrix)
            node_3 = self.find_node_by_id(node_id_3, self.nodeMatrix)
    
            airfoilElementMatrix.append([node_1, node_3, None])
    
        return airfoilElementMatrix

    def createWingElementMatrix(self):
        
        # Initialize an empty list for wing elements
        elementsWing = []

        # Helper function to find a Node's new global index after replication across airfoils
        def get_global_node(node_id, airfoil_index, nodes_per_airfoil):
            # Adjust node_id for zero-based indexing and calculate global index
            return (node_id - 1) + airfoil_index * nodes_per_airfoil

        nodes_per_airfoil = len(self.nodeMatrix) // self.numberOfAirfoils  # Assuming evenly distributed nodes

        # Iterate through airfoils to replicate elements
        for airfoil_index in range(self.numberOfAirfoils):
            for i, element in enumerate(self.elementsAirfoil):
                node_id_1, node_id_2, _ = element

                # Calculate global node IDs
                global_node_id_1 = get_global_node(node_id_1, airfoil_index, nodes_per_airfoil)
                global_node_id_2 = get_global_node(node_id_2, airfoil_index, nodes_per_airfoil)

                # Determine the element kind based on its index within the airfoil
                # Assuming the first 12 elements are of one kind (e.g., beams) and the rest are another (e.g., bars)
                kind = 2 if i < 12 else 1
            
                # Append new element with global node IDs and original kind
                elementsWing.append([global_node_id_1, global_node_id_2, kind])

        # Now, elementsWing contains global indices for start and end nodes of each element, and the element type
        # To build the wingElementMatrix with Node objects, map global indices back to Node objects
        wingElementMatrix = []
        
        for global_node_id_1, global_node_id_2, kind in elementsWing:
            start_node = self.nodeMatrix[global_node_id_1]  # Directly use global index to access node
            end_node = self.nodeMatrix[global_node_id_2]
            # Append the Node objects and kind to the matrix
            wingElementMatrix.append([start_node, end_node, kind])

        return wingElementMatrix


     
    
# Function to find a Node in nodeMatrix by node_id
    def find_node_by_id(self, node_id, nodeMatrix):
        for node in nodeMatrix:
            if node.node_id == node_id:
                return node
        return None  # Return None if no matching node is found