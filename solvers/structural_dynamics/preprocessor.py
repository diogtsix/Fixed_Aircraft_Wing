import numpy as np
from dataobjects.naca_airfoil_generation import naca2D
from dataobjects.material import Material

class Preprocessor():
    
    def __init__(self, chordLength = 1.12, wing_length = 3.61, 
                 elementMaterial = Material, elementSurface = 0.0031):
        
        self.chordLength = chordLength
        self.wing_length = wing_length
        self.elementMaterial = elementMaterial
        self.elementSurface = elementSurface
        
        self.nodeMatrix = self.createNodeMatrix()
        self.elementMatrix = self.createElementMatrix()
        
        self.elementMaterial = Material()
        
        nodes_airfoil, elements_airfoil = naca2D(chordLength, plotAirfoil = False)
        
        self.nodesAirfoil = nodes_airfoil
        self.elementsAirfoil = elements_airfoil
        
        
    
    def createNodeMatrix(self):
        pass
    
    def createElementMatrix(self):
        pass