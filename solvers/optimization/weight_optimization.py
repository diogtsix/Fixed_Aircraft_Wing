
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver
from solvers.structural_dynamics.postprocess import Postprocess
from solvers.optimization.material_database import generate_material_np_matrix


from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd 
# import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt 

from pyswarm  import pso
# DEAP library for Genetic Algorithm
from deap import base, creator, tools, algorithms

import copy
import random

class Weight_Optimization():
    
    def __init__(self, solverType = 'genetic', 
                 preprocessorObj = None, solverObj = None, postprocessorObj = None, 
                optimize_with_surrogate_ML_model = False, 
                discrete_surface_values = [0.0025, 0.0031, 0.0041, 0.0071, 0.0091, 0.011, 0.0013]):
        
        # Import material database
        # In order to add in our current model more materials in the database, we need to parametrize some 
        # parts of this class because the material_ids selection [0, 1, 2] are hardcoded at the moment.
        self.material_data_base = generate_material_np_matrix()
        
       # Set Wing Objects from structural dynamics model
        material = self.get_material('Steel')
        if preprocessorObj == None :
            self.initial_preprocessor = Preprocessor(elementMaterial = material)
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
       
                
        self.preprocessor = Preprocessor(elementMaterial = material)
        
        # For now We set as allowbable Stress the static values for all elements = Aluminium
        V_static_for_initial_Wing = self.extract_structural_attribute_array(self.initial_solver.static_structural_properties, 
                                                                       'sx')
        
        self.allowableStress = V_static_for_initial_Wing*(3/4) # (3/4) is arbitary
        self.stress_factor = max(np.abs(self.allowableStress)) # Factor for normalization
        self.allowableStress = self.allowableStress/self.stress_factor

        self.discrete_surface_values = discrete_surface_values
        self.surface_factor = 1 # Surface factor for normalization current = 1 which means we don't use it 
        self.results = None 
        self.surrogated_model = None 
        self.optimization_algorithm = solverType
        self.optimize_with_surrogate_ML_model = optimize_with_surrogate_ML_model

    # Objective Function
    def objective_function(self,opt_vars):
        
        """
            This is the objective function of the total weight calculations. 
            Inputs :optimization variables 
            Output : Weight of the wing
            
        """

        # opt_vars = np.array(x_continuous + x_integer)
        elementMatrix = self.initial_preprocessor.elementMatrix
        
        #Update the elementMatrix based on the opt_vars
        updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(opt_vars, elementMatrix)
        
        #Calc total_weight which should be the output of the objective function
        weight = self.calculate_total_weight(updated_elementMatrix)
        
        
        if self.optimization_algorithm == 'genetic':  
            c = self.stress_constraint(opt_vars)     
            if any(ci < 0 for ci in c):
                
                    penalty = 1e4 * np.abs(min(c))
                    
                    total_weight = weight + penalty

            else:
                    penalty = 0
                
            total_weight = (total_weight,)   
            
            print(f'Real Weight = {weight} || Weight with Penalty = {total_weight} || Constraint Minumum = {min(c)} || Penalty Value = {penalty}') 
                
        else:
            total_weight = weight
            
            print(f'Weight = {total_weight}')
        
  
            
        return total_weight

    # Constraint Functions
    def stress_constraint(self, opt_vars):
        """
            This is the stress constraint function. The allowble stress is set inside the constructor of 
            the weight optimization class. Here we calculate the stresses through the wing based on the otpimization 
            variable vector and we compare the solution. 
            The default value for the allowable stress is (3/4) Vstatic of the model. 
        """
        
        elementMatrix = self.initial_preprocessor.elementMatrix
        
        #Update the elementMatrix based on the opt_vars
        updated_elementMatrix = self.replace_optimized_vars_to_elementMatrix(opt_vars, elementMatrix)
        
        #Update the preprocessor based on the opt_vars
        self.preprocessor.elementMatrix = updated_elementMatrix
        
        solver = Solver(preprocessor = self.preprocessor)
        
        structuralProperties = solver.static_structural_properties
        
        stress = self.extract_structural_attribute_array(structuralProperties, 
                                                                       'sx')
              
        stress_diff = np.abs(self.allowableStress) - np.abs(stress)/self.stress_factor     
        
        if self.optimization_algorithm == 'pso':
            c =    min(stress_diff)
            print(f' Constraint Minimum Value = {c} ')
            
        elif self.optimization_algorithm == 'genetic': 
            
            c =      stress_diff                                      

        return stress_diff


    # Main Optimization Function
    def run_optimization(self):

        """
            In this function we Generate Data for ML training, we Train the Neural Network,
            and after that we run the optimization using the surrogated machine learning model. 
        """
        # Statement in case we want to run with ML surrogated model or we want to run with original Objective Function
        if self.optimize_with_surrogate_ML_model == True:
            
            numberOfSamples= 30000 # 35000 might be better to avoid overfitting
            
            filename = f'dataset_sampleNumber_{numberOfSamples}.csv'
            file_path = os.path.join(os.getcwd()+ '\datasets', filename)

                # Check if the file exists
            if os.path.exists(file_path):
                
                data = pd.read_csv(file_path)
            else:
                data = self.generate_data(numberOfSamples = numberOfSamples, 
                                    file_path = file_path)

            label = ['y']
            
            ML_model = 'Neural_Network'
            model_path = os.path.join(os.getcwd(), 'ML_models', f'{ML_model}_{numberOfSamples}.keras')
            
            
            if ML_model ==  'Neural_Network':
                
                
                # Neural Network
                X = data.drop('y', axis=1)
                y = data['y']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                n_features = X_train.shape[1]
                    
                
                if os.path.exists(model_path):
                    model = keras.models.load_model(model_path)
                    
                    model.compile(optimizer = 'adam', 
                            loss = 'mean_squared_error',
                            metrics = ['mean_absolute_error']
                            )
                                    
                    print("Loaded existing model.")

                else:

                
                    model = keras.Sequential([
                        keras.Input(shape = (n_features,)),
                        keras.layers.Dense(500, activation='relu'),
                        keras.layers.Dense(250, activation='relu'),  # First hidden layer with 128 neurons
                        keras.layers.Dense(15, activation='relu'),  # Second hidden layer with 64 neurons
                        keras.layers.Dense(1)  # Output layer for regression
                    ])
                    
                    model.compile(optimizer = 'adam', 
                                loss = 'mean_squared_error',
                                metrics = ['mean_absolute_error']
                                )

                    model.fit(X_train, y_train, epochs=125, batch_size=32, validation_split=0.1)

                    model.save(model_path)
                    
                self.surrogated_model = model 
 
        ## Run optimization   
        num_of_vars = 2 * len(self.preprocessor.elementMatrix)         
        if self.optimization_algorithm == 'pso' : 

            self.particle_swarm_optimization_method(num_of_vars)
                
        elif self.optimization_algorithm == 'genetic':
            self.genetic_algorithm_method(num_of_vars)

    
    def genetic_algorithm_method(self, n_features):
        
        """
            Genetic Algorithm Optimization Method is working properly. 
            Here is the implementation using DEAP library. 
        """
        
        HALF_FEATURES = n_features // 2 
        
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
    

        
        # Attribute generators
        toolbox.register("attr_surfaces", random.choice, self.discrete_surface_values)  # For continuous features (0.001, 0.01)

        toolbox.register("attr_int", random.choice, [0, 1 ,2])  # For discrete features

        
        # Structure initializers        
        toolbox.register("individual", tools.initIterate, creator.Individual,
                        lambda: [toolbox.attr_surfaces() for _ in range(HALF_FEATURES)] +
                                [toolbox.attr_int() for _ in range(HALF_FEATURES)])
        
        
        # Register genetic operators
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.7)  # Uniform crossover
        toolbox.register("mutate", self.custom_mutation, indpb = 1, allowed_values=self.discrete_surface_values)
        
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        
        
       
        if self.optimize_with_surrogate_ML_model == True:
            toolbox.register("evaluate", self.surrogated_model_objective_function)
        else: 
            toolbox.register("evaluate", self.objective_function)
                
        
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)  # Only the best individual kept

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, 
                                    stats=stats, halloffame=hof, verbose=True)
        
        print("Best individual is: ", hof[0], "with fitness: ", hof[0].fitness)
        
        
        final_population = pop
        best_individual = hof.items[0]
        logbook = log
        
        #Store Results
        self.results = {
            "final_population": final_population,   
            "best_individual": {
                "genotype": list(best_individual),
                "fitness": best_individual.fitness.values
            },
            "statistics": {
                "generations": log.select("gen"),
                "average_fitness": log.select("avg"),
                "min_fitness": log.select("min"),
                "max_fitness": log.select("max")
            },
            "logbook": logbook
        }

    
    def particle_swarm_optimization_method(self, n_features ):
            """
            Particle Swarm Optimization Algorithm is implemented in this function. 
            CUrrently the implementation is not working properly, and the parameters
            needs to be tuned.
            """
        
            half_point = n_features // 2
            lb = np.array([0.0025] * half_point + [0] * (n_features - half_point))
            ub = np.array([0.011] * half_point + [2] * (n_features - half_point))
            
            kwargs = {
                'swarmsize': 250,          # Increase the swarm size for better exploration default value = 100
                'omega': 1.2,               # Increase inertia weight to allow greater velocity default value = 0,5
                'phip': 0.6,              # Increase cognitive coefficient default value = 0.5
                'phig': 1,              # Increase social coefficient default value = 0.5
                'maxiter': 100,          # Maximum number of iterations default value = 100
                'debug': True,            # Enable debugging to track process default value = True
                'minstep': 1e-4,          # Set a smaller step size for termination to allow finer exploration default value = 1e-8
                'minfunc': 1-1,           # Minimum change in function value for termination default value = 1e-8
            }
            
            if self.optimize_with_surrogate_ML_model == True:
                xopt, fopt = pso(self.surrogated_model_objective_function, lb, ub, f_ieqcons=self.stress_constraint, **kwargs)
            else: 
                xopt, fopt = pso(self.objective_function, lb, ub, f_ieqcons=self.stress_constraint, **kwargs)

            print("Optimal solution:", xopt)
            print("Optimal objective value:", fopt)
            
            self.results = [xopt , fopt]
            
        
    def surrogated_model_objective_function(self, x):
        """
        This function represents the objective function for the surrogate ML model optimization method. 
        """
        
        if self.optimization_algorithm == 'pso':
        
            x_discrete = np.copy(x)
            x_discrete[271:] = np.clip(np.round(x_discrete[271:]), 0, 2)
    
            weight = self.surrogated_model.predict([np.array([x_discrete])])
            objValue = weight
            
        elif self.optimization_algorithm == 'genetic':
            
            c = self.stress_constraint(x)
            
            if any(ci <= 0 for ci in c):
                penalty = 1e4
            else:
                penalty = 0
                print(penalty)
                
            weight = self.surrogated_model.predict([np.array([x])])    
            objValue = weight+ penalty
        
        
        print(weight)
            
        
    
        
        
        
        return objValue
    
         
    def generate_data(self, numberOfSamples, file_path):
        """
        This method is used for data generation to train the ML model. 
        We use random values for surface and material ids.
        At the end of this function a .csv file is saved for latter use when numberOfSamples remains the same. 
        """
        elementMatrix = self.initial_solver.preprocessor.elementMatrix
        mat_size = len(elementMatrix)
        
        # Calculate sample sizes for each group
        num_increasing_samples = int(0.15 * numberOfSamples)
        num_random_samples = numberOfSamples - num_increasing_samples
        
        first_half_random = np.random.uniform(low=0.0005, high=0.018, size=(mat_size, num_random_samples))

        
        second_half_random = np.random.choice([0, 1, 2], size=(mat_size, num_random_samples))
        
        # Generate increasing samples (15%)
        step_size = (0.01 - 0.001) / num_increasing_samples
        first_half_increasing = np.tile(np.arange(0.0005, 0.018, step_size)[:num_increasing_samples], (mat_size, 1))
        second_half_increasing = np.tile(np.arange(0, num_increasing_samples) % 3, (mat_size, 1))

        # Combine the two parts
        first_half = np.hstack((first_half_random, first_half_increasing))
        second_half = np.hstack((second_half_random, second_half_increasing))

        self.surface_factor = np.max(first_half)
        self.surface_factor = 1 # Remove normalization
        first_half = first_half/self.surface_factor 
        
        indices = np.random.permutation(numberOfSamples)
        first_half = first_half[:, indices]
        second_half = second_half[:, indices]
        

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
        
        data_array = np.concatenate([samples.T, y_measured_reshaped], axis=1)
        data = pd.DataFrame(data_array, columns=features + ['y'])
        
        # Export dataset
        current_directory = os.getcwd()
        data.to_csv(file_path, index=False)
        return data
        
        
    def calculate_total_weight(self, elementMatrix):
        """
        Given as an input the element Matrix, we calculate the total weight of the wing.
        """
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
                return row[2]
        raise ValueError(f"Material '{material_name}' not found in database.")
    
    
    def extract_optimization_vars_from_elementMatrix(self, elementMatrix_input):
        """
        This class method is for extracting the vector of [surfaces materials] from them elementsMatrix.
        This function is used for comparison purposes
        """
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
        
        updated_elementMatrix = self.manual_deep_copy_element_matrix(elementMatrix) # ElementMatrix should be a list or a type that supports .copy

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
        FUnction to extract the materil object from my database. 
        """

        
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
    
    
    def manual_deep_copy_element_matrix(self, elementMatrix):
        
        new_matrix = []
        for element in elementMatrix:
            # Assuming each element is a list [start_node, end_node, material, surface, other_info...]
            # Deep copy each component as necessary
            new_element = [
                copy.deepcopy(element[0]),  # deep copy if it's a custom object or if needed
                copy.deepcopy(element[1]),
                element[2],  # Assume material object needs proper handling
                copy.deepcopy(element[3]),  # Assuming this is a scalar like a float or int, no need to deep copy
                element[4],
                element[5]
            ]
            new_matrix.append(new_element)
        return new_matrix
    
    def custom_mutation(self, individual, indpb, allowed_values):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.choice(allowed_values)
        return individual,
    
    
    
def main():   
    a = Weight_Optimization()
    a.run_optimization()
    
if __name__ == "__main__":
    main()