import numpy as np
import matplotlib.pyplot as plt


def naca2D(chordLength, plotAirfoil = False):
    # Define the airfoil nodes
    points = np.array([
        [1.000000, -0.000000],
        [0.917284,  0.018842],
        [0.834388,  0.035845],
        [0.751335,  0.051152],
        [0.668142,  0.064790],
        [0.584823,  0.076659],
        [0.501392,  0.086518],
        [0.417862,  0.093954],
        [0.334251,  0.098330],
        [0.250587,  0.098652],
        [0.166908,  0.093233],
        [0.079453,  0.077656],
        [0.000000,  0.000000],
        [0.087214, -0.038767],
        [0.166425, -0.053452],
        [0.249413, -0.059763],
        [0.332415, -0.061018],
        [0.415471, -0.058906],
        [0.498608, -0.054419],
        [0.581843, -0.048195],
        [0.665191, -0.040647],
        [0.748665, -0.032017],
        [0.832279, -0.022402],
        [0.916049, -0.011778],
        [1.000000,  0.000000]
    ])
    
    if points.shape[0] != 25:
        raise ValueError('You did not choose 24 points, go again to NACA URL')
    
    # Remove the last row as it is a duplicate of the first
    points = points[:-1, :]
    
    # Scale X and Y coordinates
    L_chord = chordLength
    X = points[:, 0] * L_chord
    Y = points[:, 1] * L_chord
    
    # Prepare nodes_airfoil
    nodes_airfoil_indices = [12, 14, 10, 16, 8, 18, 6, 20, 4, 22, 2, 0]
    nodes_airfoil = np.zeros((12, 2))
    for i, idx in enumerate(nodes_airfoil_indices):
        nodes_airfoil[i, 0] = X[idx]
        nodes_airfoil[i, 1] = Y[idx]
    
    # Prepare elements_airfoil
    elements_airfoil = np.array([
        [1, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 12], [12, 10], [10, 8], [8, 6], [6, 4], [4, 2],
        [2, 1], [2, 3], [2, 5], [4, 5], [5, 6], [6, 7], [6, 9], [8, 9], [8, 11], [10, 11]
    ], dtype=int)
    
    # Adding a third column to elements_airfoil for compatibility with MATLAB format (set to zeros)
    elements_airfoil = np.hstack((elements_airfoil, np.zeros((elements_airfoil.shape[0], 1), dtype=int)))
    
    if plotAirfoil == True :
        plot_airfoil(X, Y)
        
    return nodes_airfoil, elements_airfoil



def plot_airfoil(X, Y):
    """
    Plots the airfoil by plotting every other point and labeling it.
    
    Parameters:
    X (array-like): X-coordinates of the airfoil points.
    Y (array-like): Y-coordinates of the airfoil points.
    """
    plt.figure(figsize=(10, 6))  # Create a new figure with a specific size
    plt.axis('equal')  # Set equal scaling by aspect ratio

    for i in range(len(X)):
        if i % 2 == 0:  # Python indexing starts at 0, so this effectively selects every other point
            plt.plot(X[i], Y[i], 'o')  # Plot the point
            plt.text(X[i], Y[i], str(i + 1), fontsize=12)  # Label the point, adjusting index to match MATLAB's 1-based indexing
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Airfoil Plot')
    plt.show()

# nodes , elements = naca2D(1.12, True)

# print(nodes)
# print(elements)