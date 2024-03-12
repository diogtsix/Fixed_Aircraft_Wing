from dataobjects.barElement import BarElement
from dataobjects.naca_airfoil_generation    import naca2D
from solvers.structural_dynamics.preprocessor import Preprocessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example usage
""""
truss = TrussElement(radius=0.05, length=2.0)
print(f"Area: {truss.area}")
print(f"Surface Moment of Inertia: {truss.surfaceInertia}")
print(f"Polar Moment of Inertia: {truss.polarInertia}")

nodes_airfoil, elements_airfoil = naca2D(chordLength=1.12 , plotAirfoil = True)

print("Nodes Airfoil:\n", nodes_airfoil)
print("Elements Airfoil:\n", elements_airfoil)

"""

pre = Preprocessor()


#print(pre.elementsAirfoil)


for element in pre.elementMatrix:
    first_node = element[2]  # Get the first Node object from each row
    #print(element)
    
    
  


def visualize_wing_3d(wingElementMatrix):
    # Initialize a new figure for 3D plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Iterate through each element in the wing element matrix
    for element in wingElementMatrix:
        start_node, end_node, _ = element  # Assuming the third value is element type, which we don't use for plotting
        
        # Extract coordinates for the start and end nodes
        x_values = [start_node.coords[0], end_node.coords[0]]
        y_values = [start_node.coords[1], end_node.coords[1]]
        z_values = [start_node.coords[2], end_node.coords[2]]
        
        # Plot a line between the start and end node of each element
        ax.plot(x_values, y_values, z_values, 'b', marker='o', markersize=5, linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of the Wing')
    
    
        # Set the same range for all axes
    ax.set_xlim([-0.75, 1.3])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 4])
    
    
    plt.show()

print(pre.totalDofs)
print(pre.totalNodes)
print(pre.totalElements)

# Assuming wingElementMatrix is defined and populated as described earlier
visualize_wing_3d(pre.elementMatrix)