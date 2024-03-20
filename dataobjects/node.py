import numpy as np

class Node():
    
    def __init__(self, coords=None, dof_id=None, 
                 force=None, boundaryCondition=None, node_id = 1):
        
        """
        Initialize a Node instance.
        
        Parameters:
        coords (np.array): [X, Y, Z] coordinates in 3D space.
        dof_id (np.array): Global IDs for the degrees of freedom for each node, 
                           specifically [x_trans, y_trans, z_trans, theta_x_rot, theta_y_rot, theta_z_rot].
        force (np.array): Force or moment applied at node.
        boundaryCondition (np.array): Boundary conditions for the node.
        """
        if coords is None:
            coords = np.array([1, 1, 1])
        if dof_id is None:
            dof_id = np.array([1, 2, 3, 4, 5, 6])
        if force is None:
            force = np.array([0, 0, 0, 0, 0, 0])
        if boundaryCondition is None:
            boundaryCondition = np.array([0, 0, 0])
        
        self.coords = coords
        self.dof_id = dof_id
        self.force = force
        self.boundaryCondition = boundaryCondition
        self.node_id = node_id