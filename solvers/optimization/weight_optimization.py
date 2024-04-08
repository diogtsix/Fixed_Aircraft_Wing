
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint

from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver
from solvers.structural_dynamics.postprocess import Postprocess

from solvers.optimization.material_database import generate_material_np_matrix

from gekko import GEKKO
from gekko import ML
from gekko.ML import Gekko_NN_TF,Gekko_LinearRegression
from gekko.ML import Bootstrap,Conformist,CustomMinMaxGekkoScaler

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
import tensorflow as tf

import pandas as pd 

import matplotlib.pyplot as plt 
 
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
        self.results = None 

        
        # For now We set as allowbable Stress the static values for all elements = Aluminium
        SS = self.extract_structural_attribute_array(self.initial_solver.static_structural_properties, 
                                                                       'sx')
        
        self.allowableStress = SS*(3/4)
        self.stress_factor = max(np.abs(self.allowableStress))
        self.allowableStress = self.allowableStress/self.stress_factor


    # Objective Function
    # def objective_function(self,x_continuous, x_integer):
    def objective_function(self,opt_vars):

        # opt_vars = np.array(x_continuous + x_integer)
        elementMatrix = self.initial_preprocessor.elementMatrix
        
        #Update the elementMatrix based on the opt_vars
        updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(opt_vars, elementMatrix)
        
        #Calc total_weight which should be the output of the objective function
        total_weight = self.calculate_total_weight(updated_elementMatrix)
        
        # print(total_weight)
        
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
      
        print(np.min((np.abs(self.allowableStress) - np.abs(stress)/self.stress_factor)  ) )                                                         
        return np.abs(self.allowableStress) - np.abs(stress)/self.stress_factor

    # Main Optimization Function
    def run_optimization(self):
        
        #Solve win for initial point (static solution)
        element_matrix_initial = self.initial_solver.preprocessor.elementMatrix
        num_elements = len(element_matrix_initial) 
        
        initial_opt_vars, initi_surfaces, init_ids = self.extract_optimization_vars_from_elementMatrix(element_matrix_initial)
        
        initial_point = initial_opt_vars.flatten()  # Initial the starting point for iterations
        
        # Create the bounds for opt_vars
        bounds_first_half = [(0.1, 10) for _ in range(num_elements)] 
        bounds_second_half = [(0, 2) for _ in range(num_elements)] 
        bounds = bounds_first_half + bounds_second_half 
        

        
        constraints = [{'type': 'ineq', 'fun': self.stress_constraint}
                       ] # The opt algorithm will ensure that ineq will stay >= 0
        
        
        # constraints = [{'type': 'eq', 'fun': self.discrete_constraints}
        #                ] 
        
        if self.solverType == 'trust-constr':
            result = minimize(fun = self.objective_function, x0 = initial_point,
                            method = 'trust-constr', 
                            bounds = bounds, 
                            constraints = constraints, 
                            options={'verbose': 3, 'maxiter': 10, 
                            'initial_tr_radius': 5})
            
        elif self.solverType == 'SLSQP': 
            
            # For SLSQP
            result = minimize(fun=self.objective_function, 
                    x0=initial_point, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints, 
                    options={'ftol': 1e-6, 'eps': 1.5e-4, 'maxiter': 10})
            
        # lb =  [(-1) for _ in range(num_elements)] 
        # ub = [(np.inf) for _ in range(num_elements)] 
        # nlc = NonlinearConstraint(self.stress_constraint, lb, ub)
        
        # result = differential_evolution(self.objective_function, bounds, constraints=nlc, 
        #                                 x0 = initial_point, maxiter = 10, disp = True, 
        #                                 workers = 1, updating = 'immediate', 
        #                                 mutation = 0.001)
 
        
        # Remove normalization from the surfaces 
        num_of_element = len(element_matrix_initial)
        
        result.x[:num_of_element] = result.x[:num_of_element] * self.surface_factor
        
        print("Objective Function Value:", result.fun)
        print("Optimization Variables:", result.x)
        print("Optimization Result:", result)
         
        self.results = result
 
    def run_optimization_ML(self):
        
        
        gekkoObj = GEKKO()
        
        y_measured, xl, features = self.generate_data(numberOfSamples = 10000)

        data_array = np.concatenate([xl.T, y_measured], axis=1)
        data = pd.DataFrame(data_array, columns=features + ['y'])
        label = ['y']
        
        ML_model = 'Neural_Network'
        
        
        
        if ML_model == 'SVR' :
        
            #SVR
            train , test = train_test_split(data,test_size=0.2,shuffle=True)
            svr = svm.SVR()
            svr.fit(train[features],np.ravel(train[label]))
            r2 = svr.score(test[features],np.ravel(test[label]))
            print('svr r2:',r2)
        
        elif ML_model ==  'Neural_Network':
            # Neural Network
            X = data.drop('y', axis=1)
            y = data['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            n_features = X_train.shape[1]
           
            model = keras.Sequential([
                keras.Input(shape = (n_features,)),
                keras.layers.Dense(25, activation='relu'),  # First hidden layer with 128 neurons
                keras.layers.Dense(25, activation='relu'),  # Second hidden layer with 64 neurons
                keras.layers.Dense(1)  # Output layer for regression
            ])
            
            model.compile(optimizer = 'adam', 
                        loss = 'mean_squared_error',
                        metrics = ['mean_absolute_error']
                        )

            model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1)

            test_loss, test_mae = model.evaluate(X_test, y_test)
            print(f"Test Loss (MSE): {test_loss}, Test MAE: {test_mae}")

            y_pred = model.predict(X_test)
            
            
            # Scatter plot of actual vs. predicted values
            plt.scatter(y_test, y_pred, alpha=0.5)

            # Line for perfect predictions
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs. Predicted Values')
            plt.show()

        
        
        
    
    
    def generate_data(self, numberOfSamples):
        
        elementMatrix = self.initial_solver.preprocessor.elementMatrix
        
        mat_size = len(elementMatrix)
        
        first_half = np.random.uniform(low=0.001, high=0.01, size=(mat_size, numberOfSamples))
        self.surface_factor = np.max(first_half)
        first_half = first_half/self.surface_factor 
        
        second_half = np.random.choice([0, 1, 2], size=(mat_size, numberOfSamples))

        samples = np.vstack((first_half, second_half))
        
        # Generate the "features" array vector
        features_first_half = [f"A{i+1}" for i in range(mat_size)]
        features_second_half = [f"material{i+1}" for i in range(mat_size)]
        features = features_first_half + features_second_half

        #replace Samples into the elementMatrix
        y_measured = np.zeros(numberOfSamples)
        
        for i in range(numberOfSamples):
            
            sample = samples[:,i]
            
            updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(sample, elementMatrix)
            y_measured[i] = self.calculate_total_weight(updated_elementMatrix)
        
        y_measured_reshaped = y_measured.reshape(-1, 1)  # Makes y_measured a 2D array with a single colum
        
        return y_measured_reshaped, samples, features 
    
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
        
        num_elements = len(elementMatrix)
        
        elementMatrix_np = np.array(elementMatrix, dtype=object)
        
        surfaces = elementMatrix_np[:, 4].astype(float).reshape(-1, 1) 
        
               
        material_objects = elementMatrix_np[:, 3] 
        
        material_ids = np.array([obj.id for obj in material_objects], dtype=int).reshape(-1, 1)
        
        # material_ids = np.random.choice([0, 1, 2], size=num_elements).reshape(-1,1)
        
        surfaces = surfaces.reshape(-1, 1)
        
        num_elements = len(elementMatrix)
        # surfaces = np.random.uniform(low=0.002, high=0.01, size=num_elements).reshape(-1,1)
        
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
            val_surface = surface_areas[i]
            element[4] = val_surface * self.surface_factor

            # Lookup and update material object
        
            val_id = material_ids[i]
            
            val_id =  round(val_id)# WHen I add also materila optimization this line will removed
            material_id = val_id
            material_object = self.get_material_by_id(material_id)
            element[3] = material_object


        return updated_elementMatrix
                       
            
    def get_material_by_id(self, material_id):
        """
        FUnction to extract the materil object from my database
        """
        
        # material_id = round(material_id, 4)
        
        if material_id <= 0 :
            material_id = 0 
        elif material_id >=2 : 
            material_id = 2
        else:
            material_id = round(material_id)
            
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
        '''
        This constraint is added to avoid using integer variables in our solver. 
        By implementing the following equation and equating it with zero
        the constraint must be satisfied. If the constraint is satisfied then we have predermined values for our materials
        '''
        # elementMatrix = self.preprocessor.elementMatrix
        # num_vars = len(opt_vars) // 2 
        
        # material_ids = opt_vars[num_vars:]
        
        # Z_element = np.zeros([num_vars,1 ])
        # for index, id in enumerate(material_ids):
            
        #     material_id = id
        #     P = 1
        #     for im in range(len(self.material_data_base)):
                
        #         P = P * (material_id - self.material_data_base[im][1])
            
        #     Z_element[index] = P 
        # Z_surfaces = np.zeros([num_vars])    
        
        # combined_vector = np.concatenate((Z_surfaces, Z_element.flatten()))
        # print('discrete')
        # return combined_vector
        
        elementMatrix = self.preprocessor.elementMatrix
        num_vars = len(opt_vars) // 2 
        
        material_ids = opt_vars[num_vars:]
        
   
        Z_element = np.zeros([num_vars,1 ])
        for index, id in enumerate(material_ids):
            
            material = self.get_material_by_id(id)

            Mulitplier = 1
            Summation = 0
            for im in range(len(self.material_data_base)):
                
                
                     
                a = vars(material)
                b = vars(self.material_data_base[im][2])
                    
                if a == b:
                    Summation = 0 
                else: 
                    Summation = 100

                
                Mulitplier = Mulitplier * (Summation)
            
            Z_element[index] = Mulitplier 
        Z_surfaces = np.zeros([num_vars])    
        
        combined_vector = np.concatenate((Z_surfaces, Z_element.flatten()))
        
        return Z_element.flatten()
    
  
        print(f"Current best solution: {intermediate_result.x}")
        print(f"Objective function value: {intermediate_result.fun}")
    
def main():   
    a = Weight_Optimization()
    # a.run_optimization()
    a.run_optimization_ML()
    
if __name__ == "__main__":
    main()