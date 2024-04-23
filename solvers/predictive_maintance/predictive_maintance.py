
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np 
import matplotlib.pyplot as plt

from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver

from dataobjects.material import Material
class Predictive_maintance():
    
    def __init__(self):
        
        """
        One block of loading is considered to represent 100 flights of 4 hours each = 400 hours each block. 
        Material is choosen = Cu-Mg Aluminum Alloys like AA2024-T351
        """
        # Properties for Material  : Cu-Mg Aluminum Alloys like AA2024-T351
        self.element_material = Material(id = 1, elasticModulus = 73100000000,
                                         poissonRatio=0.33, density = 2780,
                                         shearModulus=28000000000, 
                                         C = 7.9927, # C = 9.5 for stress units equal to MPa
                                         m = 0.235) # This is a median value for m
        
        # Load Structural Dynamics Model 
        preprocessorObj = Preprocessor(elementMaterial= self.element_material)
        self.solver = Solver(preprocessor=preprocessorObj) 
        
        self.f0 = 500
        f0 = self.f0
        self.loading_conditions_table = np.array([f0/2 , f0 , 2*f0, 3*f0, 4*f0, 5*f0, 6*f0])
        self.loading_cycles_number_table = np.array([1000, 500, 600, 100, 30 ,5, 15])
        
        self.alternation_stress_table = [] 
        self.mean_stress_table = []
        self.min_stress_table = []
        self.max_stress_table = []
        
        self.log_Nf = [] 
        self.Nf = [] 
        
        self.accumulated_damage = [] 
        self.accumulated_damage_total = None 
        
        self.crack_condition = None 
        
        self.number_of_blocks_to_failure = None 
        self.fatigue_safe_life_hours = None 
        

        
        self.fatigue_analysis()

    def fatigue_analysis(self):
        
        self.create_stress_tables()
        
        self.create_S_N_plots()
        
        self.calc_cycles()
        
        self.accumulated_damage_calc()
        
        self.miner_rule()

    def create_stress_tables(self):
                
        for index, f0 in enumerate(self.loading_conditions_table):
            preprocessorObj = Preprocessor(elementMaterial=self.element_material, forceValue= f0)
            solverObj = Solver(preprocessor = preprocessorObj)
            solverObj.solve_with_eigenAnalysis()
            
            structuralProperties = solverObj.structural_properties

            min_sx, max_sx = self.extract_min_max_sx(structuralProperties)

            self.min_stress_table.append(min_sx)
            self.max_stress_table.append(max_sx)
            
            mean_sx = (min_sx + max_sx) / 2
            alt_sx = (max_sx - min_sx) / 2
            
            self.mean_stress_table.append( mean_sx)
            self.alternation_stress_table.append(alt_sx)
    
    def create_S_N_plots(self):
        
                
        min_val = min(self.alternation_stress_table) / 1e6
        max_val = max(self.alternation_stress_table) / 1e6
          
        stress_amp = np.linspace(min_val/2, 2*max_val, num = 1000)
        
        cycles_to_failure = self.calc_cycles_to_failure(stress_amp, stress_mean=0)
        
        
        plt.figure()
        plt.plot( cycles_to_failure, stress_amp,  label = "AA 7075")
        plt.ylabel('Alternating Stress (MPa)')
        plt.xlabel('Cycles to Failure (N)')
        plt.title('S-N Curve for Aluminum Alloy 7075')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
                
    def calc_cycles_to_failure(self, stress_amp, stress_mean):
        
        C = self.element_material.C
        m = self.element_material.m


        cycles_to_failure = 10**(C-m * np.log10(stress_amp))
        
        # cycles_to_failure = 10**(C - 1.11862*np.log10(stress_mean) - m*stress_amp + 5*1e-3*(stress_amp**2))
        
        
        return cycles_to_failure
               
    def extract_min_max_sx(self, Structural_properties):
        
        max_sx = float('-inf')
        min_sx = float('inf')  

        for row in Structural_properties:
            for obj in row:
                if obj.sx > max_sx:
                    max_sx = obj.sx
                    
                    

        for row in Structural_properties:
            for obj in row[-10:]:  # Only consider the last 10 columns
                if obj.sx < min_sx:
                    min_sx = obj.sx
            

        
        return min_sx, max_sx
    
    def calc_cycles(self):
        
        for index, stress in enumerate(self.alternation_stress_table):
            
            stress_mean = self.mean_stress_table[index] / 1e6
            
            cycle =  self.calc_cycles_to_failure(stress_amp= stress/1e6 , 
                                                 stress_mean = stress_mean / 1e6)
            
            if cycle < 5*1e6 : 
                self.Nf.append(cycle)
            else :
                self.Nf.append('inf')   
                
            self.log_Nf.append(np.log10(cycle))    
    
    def accumulated_damage_calc(self):
        
        for index, condition in enumerate(self.loading_cycles_number_table):
            
            if not self.Nf[index] == 'inf':
                
                val = condition/self.Nf[index]
                self.accumulated_damage.append(val)
                
                self.accumulated_damage_total = self.accumulated_damage_total + val
            
        if self.accumulated_damage_total < 1: 
            self.crack_condition = False
        else:
            self.crack_condition = True
   
    def miner_rule(self):
        
        self.number_of_blocks_to_failure = 1/ self.accumulated_damage_total
        self.fatigue_safe_life_hours = 400 * self.number_of_blocks_to_failure
                        
def main():
    
    a = Predictive_maintance()
    a = 2

if __name__== "__main__":
    main()