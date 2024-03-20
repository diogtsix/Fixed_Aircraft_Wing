from dataobjects.barElement import BarElement
from dataobjects.naca_airfoil_generation    import naca2D
from solvers.structural_dynamics.preprocessor import Preprocessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solvers.structural_dynamics.solver import Solver
import numpy as np

from solvers.structural_dynamics.postprocess import Postprocess



pre = Preprocessor(numberOfAirfoils= 11)

solve = Solver(preprocessor= pre, timeStep= 0.005, simulationTime= 0.8)
solve.solve_with_eigenAnalysis()

post = Postprocess(solver=solve)

#post.plotEigenModes(numberOfModes = 8, scaling_factor = 2)
#post.simulation_displacements(scaling_factor= 3)

post.frequencyResponse()


