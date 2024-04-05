
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from scipy.optimize import minimize

from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver
from solvers.structural_dynamics.postprocess import Postprocess

from solvers.optimization.material_database import generate_material_np_matrix

from gekko import GEKKO
 
class Weight_Optimization():
    
    def __init__(self, solverType = 'SLSQP', preprocessorObj = None, solverObj = None, 
                postprocessorObj = None):
        
        self.material_data_base = generate_material_np_matrix()
       # Set Wing Objects from structural dynamics model
        if preprocessorObj == None :
            self.initial_preprocessor = Preprocessor(elementMaterial = self.get_material('Aluminum'))
        else:
            self.initial_preprocessor = preprocessorObj
        if solverObj == None:
            self.initial_solver = Solver(preprocessor = self.initial_preprocessor)
        else:
            self.initial_solver = solverObj
        if postprocessorObj ==None:
            self.initial_postprocessor = Postprocess(solver = self.initial_solver)
        else:
            self.initial_postprocessor = postprocessorObj
       
        
        self.solverType = solverType
        
        self.preprocessor = self.initial_preprocessor
        self.solver = self.initial_solver
        self.initial_postprocessor = self.initial_postprocessor
        self.surface_factor = None 

        
        # For now We set as allowbable Stress the static values for all elements = Aluminium
        self.allowableStress = self.extract_structural_attribute_array(self.initial_solver.static_structural_properties, 
                                                                       'sx')


    # Objective Function
    # def objective_function(self,x_continuous, x_integer):
    def objective_function(self,opt_vars):

        # opt_vars = np.array(x_continuous + x_integer)
        elementMatrix = self.initial_preprocessor.elementMatrix
        
        #Update the elementMatrix based on the opt_vars
        updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(opt_vars, elementMatrix)
        
        #Calc total_weight which should be the output of the objective function
        total_weight = self.calculate_total_weight(updated_elementMatrix)
        
        print(total_weight)
        
        return total_weight

    # Constraint Functions
    # def stress_constraint(self, x_continuous, x_integer):
    def stress_constraint(self, opt_vars):
       
       
        # opt_vars = np.array(x_continuous + x_integer)
        
        elementMatrix = self.initial_preprocessor.elementMatrix
        
        #Update the elementMatrix based on the opt_vars
        updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(opt_vars, elementMatrix)
        
        #Update the preprocessor based on the opt_vars
        self.preprocessor.elementMatrix = updated_elementMatrix
        
        solver = Solver(preprocessor = self.preprocessor)
        
        structuralProperties = solver.static_structural_properties
        
        stress = self.extract_structural_attribute_array(structuralProperties, 
                                                                       'sx')
        a = 2 
        print(a)                                                            
        return np.abs(self.allowableStress) - np.abs(stress)

    # Main Optimization Function
    def run_optimization(self):
        
        #Solve win for initial point (static solution)
        element_matrix_initial = self.initial_solver.preprocessor.elementMatrix
        
        initial_opt_vars, initi_surfaces, init_ids = self.extract_optimization_vars_from_elementMatrix(element_matrix_initial)
        
        initial_point = initial_opt_vars.flatten()  # Initial the starting point for iterations
        
        # constraints = [{'type': 'eq', 'fun': self.discrete_constraints}, 
        #                {'type': 'ineq', 'fun': self.stress_constraint}
        #                ] # The opt algorithm will ensure that ineq will stay >= 0
        constraints = [{'type': 'eq', 'fun': self.discrete_constraints}
                       ] # The opt algorithm will ensure that ineq will stay >= 0
            
        options = [{'maxiter': 5, 'disp': True}]
        
        result = minimize(fun = self.objective_function, x0 = initial_point,
                          method = 'trust-constr',  constraints = constraints, 
                          options={'verbose': 3, 'maxiter': 100})
        
        
        # Remove normalization from the surfaces 
        num_of_element = len(element_matrix_initial)
        
        result.x[:num_of_element] = result.x[:num_of_element] * self.surface_factor
        
        print("Objective Function Value:", result.fun)
        print("Optimization Variables:", result.x)
        print("Optimization Result:", result)

 
       
        
    def material_constraint(x):
        # Assuming x is an array where the first half represents surfaces
        # and the second half represents material IDs
        material_ids = [1, 2, 3] 
        num_elements = len(x) // 2  # Assuming x is structured as [surfaces, material_ids]
        material_vars = x[num_elements:]  # Extract material variables from x
        
        # Initialize penalty
        penalty = 0
        
        # Calculate the penalty for each material variable being away from discrete material IDs
        for m_var in material_vars:
            # Calculate the minimum distance to the nearest material ID
            min_distance = min(abs(m_var - mid) for mid in material_ids)
            
            penalty += min_distance**30
            print(material_vars)
            print(penalty)
        return -penalty
       
        
    def calculate_total_weight(self, elementMatrix):
        
        total_weight = 0
        
        for element in elementMatrix:
            
            start_node, end_node, material, surface = element[0], element[1], element[3], element[4]

            # Calculate element length
            length = np.linalg.norm(np.array(end_node.coords) - np.array(start_node.coords))
            
            # Calculate volume
            volume = length * surface
            
            # Calculate weight
            weight = volume * material.density
            
            # Add to total weight
            total_weight += weight
        
        return total_weight
    
    
    def get_material(self, material_name):
        
        for row in self.material_data_base:
            if row[0] == material_name:
                return row[1]
        raise ValueError(f"Material '{material_name}' not found in database.")
    
    
    def extract_optimization_vars_from_elementMatrix(self, elementMatrix_input):
        elementMatrix = elementMatrix_input
        
        elementMatrix_np = np.array(elementMatrix, dtype=object)
        
        surfaces = elementMatrix_np[:, 4].astype(float).reshape(-1, 1) 
        
               
        material_objects = elementMatrix_np[:, 3] 
        
        material_ids = np.array([obj.id for obj in material_objects], dtype=int).reshape(-1, 1)
        
        surfaces = surfaces.reshape(-1, 1)
        
        num_elements = len(elementMatrix)
        surfaces = np.random.uniform(low=0.002, high=0.01, size=num_elements).reshape(-1,1)
        
        # Normalize surfaces
        self.surface_factor = np.max(surfaces)
        surfaces = surfaces/max(surfaces)
        
        for ii in range (len(surfaces)):
            surfaces[ii] = surfaces[ii] 
        combined_vector = np.vstack((surfaces, material_ids))
                
        return combined_vector, surfaces, material_ids
        
                
    def replace_optimized_vars_to_elementMatrix(self, opt_vars, elementMatrix):
        
        num_elements = len(elementMatrix)
        surface_areas = opt_vars[:num_elements]  # First half is surface areas
        material_ids = opt_vars[num_elements:]  # Second half is material IDs
        
        updated_elementMatrix = elementMatrix.copy() # ElementMatrix should be a list or a type that supports .copy

        for i, element in enumerate(updated_elementMatrix):
            # Update surface area
            # val_sruface = float(str(surface_areas[i].VALUE))
            val_surface = surface_areas[i]
            element[4] = val_surface * self.surface_factor

            # Lookup and update material object
            # val_id =  int(str(material_ids[i].VALUE))
            val_id = material_ids[i]
            material_id = val_id
            material_object = self.get_material_by_id(material_id)
            element[3] = material_object


        return updated_elementMatrix
                       
            
    def get_material_by_id(self, material_id):
        """
        FUnction to extract the materil object from my database
        """
        
        material_id = round(material_id, 4)
        
        for row in self.material_data_base:
            if row[1] == material_id:
                return row[2]  # Return the Material object
        raise ValueError(f"Material ID {material_id} not found in database.")
    
    
    def extract_structural_attribute_array(self, Structural_properties, attribute_name):
        
        # Initialize an empty list to store the attribute values
        attribute_values = []

        # Iterate over each row in the structural_properties array
        for prop in Structural_properties:
            
            
            array_val  = getattr(prop[0], attribute_name, None)
            if np.isscalar(array_val):
                val = array_val
            else:
                val  = array_val[0]
            
            attribute_values.append(val)
            
        attribute_values_array = np.array(attribute_values)
        
        # Convert the list of attribute values into a NumPy array
        return attribute_values_array
    
        
    def discrete_constraints(self,opt_vars):
        
        elementMatrix = self.preprocessor.elementMatrix
        num_vars = len(opt_vars) // 2 
        
        material_ids = opt_vars[num_vars:]
        
        Z_element = np.zeros([num_vars,1 ])
        for index, id in enumerate(material_ids):
            
            material_id = id
            P = 1
            for ii in range(len(self.material_data_base)):
                
                P = P * (material_id - self.material_data_base[ii][1])
            
            Z_element[index] = P 
        Z_surfaces = np.zeros([num_vars])    
        
        combined_vector = np.concatenate((Z_surfaces, Z_element.flatten()))
        print('discrete')
        return combined_vector
        
def main():   
    a = Weight_Optimization()
    a.run_optimization()

if __name__ == "__main__":
    main()