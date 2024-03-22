
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from scipy.optimize import minimize

from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver
from solvers.structural_dynamics.postprocess import Postprocess

from solvers.optimization.material_database import generate_material_np_matrix
 
 
 
class Weight_Optimization():
    
    def __init__(self, solverType = 'L-BFGS-B'):
        
       self.material_data_base = generate_material_np_matrix()

       self.initial_preprocessor = Preprocessor(elementMaterial = self.get_material('Aluminum'))
       self.initial_solver = Solver(preprocessor = self.initial_preprocessor)
       self.initial_postprocessor = Postprocess(solver = self.initial_solver)
       
       self.allowableStress = None
       self.solverType = solverType
       

    # Objective Function
    def objective_function(self,design_vars):
        # Calculate total weight based on design_vars
        # design_vars could include indices for material choices and cross-sectional areas
        design_vars = 1
        total_weight = self.calculate_total_weight(design_vars)
        return total_weight

    # Constraint Functions
    def stress_constraint(self,design_vars):
        # Placeholder for stress constraint calculation
        max_stress = self.calculate_max_stress(design_vars)
        return self.allowable_stress - max_stress

    # Main Optimization Function
    def run_optimization(self):
        
        initial_point = [0, 0.005]  # Example: [Material index, cross-sectional area]
        
        constraints = [{'type': 'ineq', 'fun': self.stress_constraint}]  # Define other constraints similarly
        
        result = minimize(fun = self.objective_function, x0 = initial_point,
                          method = self.solverType,  constraints = constraints)
        print("Optimization Result:", result)
        
    def calculate_total_weight(self, design_vars):
        design_vars = 1
    
    
    def get_material(self, material_name):
        
        for row in self.material_data_base:
            if row[0] == material_name:
                return row[1]
        raise ValueError(f"Material '{material_name}' not found in database.")
    

def main():   
    a = Weight_Optimization()
    print(a)

if __name__ == "__main__":
    main()