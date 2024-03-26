import numpy as np
from solvers.structural_dynamics.preprocessor import Preprocessor
from dataobjects.barElement import BarElement
from dataobjects.beamElement import BeamElement
from scipy.linalg import eigh
from Utilities.newmark import Newmark
from Utilities.eigenAnalysis import eigenAnalysis
from dataobjects.structural_properties import Structural_Properties

class Solver():
    
    def __init__(self, preprocessor = Preprocessor(), 
                 barElement = BarElement(radius = 0.0316), 
                 beamElement = BeamElement(radius = 0.0316), 
                 simulationTime = 5, timeStep = 0.05):
        
        self.preprocessor = preprocessor
        
        self.stiffnessFullMatrix = np.zeros([self.preprocessor.totalDofs, self.preprocessor.totalDofs])
        self.massFullMatrix = np.zeros([self.preprocessor.totalDofs, self.preprocessor.totalDofs])
        self.forceVector = np.zeros([self.preprocessor.totalDofs, 1])
        
        self.barElement = barElement
        self.beamElement = beamElement
        
        self.K = None # Stiffness matrix
        self.M = None # Mass matrix
        self.F = None # Force Vector
        self.C = None # Damping matrix
        self.eigenModes = None 
        self.eigenfrequencies = None 
        self.staticDisplacement =  None 
        self.timeStep = timeStep
        self.simulationTime = simulationTime
        
        self.t_Newmark = None  # time
        self.x_Newmark = None # Displacement
        self.dx_Newmark = None # Velocity
        self.ddx_Newmark = None # Acceleration
        
        self.t_eigenAnalysis = None 
        self.x_eigenAnalysis = None
        
        self.structural_properties = None
        self.static_structural_properties = None
        
        self.createGlobalMatrices()
        
        self.eigenAnalysis()
        
        self.RayleighDamping()
        
    
    def createGlobalMatrices(self):
        
        for ii in range(self.preprocessor.totalElements):
            
            element = self.preprocessor.elementMatrix[ii]
            node1 = element[0]
            node2 = element[1]
            element_type = element[2]
            
            nodeMatrix = self.preprocessor.nodeMatrix
            
            x1 = nodeMatrix[node1.node_id - 1].coords[0]
            y1 = nodeMatrix[node1.node_id - 1].coords[1]
            z1 = nodeMatrix[node1.node_id - 1].coords[2]
            x2 = nodeMatrix[node2.node_id - 1].coords[0]
            y2 = nodeMatrix[node2.node_id - 1].coords[1]
            z2 = nodeMatrix[node2.node_id - 1].coords[2] 
            
            if element_type == 1: # Bar Elements
                
                _, _, _, element_material, element_surface, _ = element
                K, M = self.KMBar3D(x1,y1,z1,x2,y2,z2,
                                    element_material, element_surface)
                
                # local X1 , localY1, localZ1, localX2, localY2, localZ2  
                LG = np.array([ node1.dof_id[0], node1.dof_id[1], node1.dof_id[2],
                               node2.dof_id[0], node2.dof_id[1], node2.dof_id[2]])
                LG = LG - 1
                num_of_dofs = 6
                
            elif element_type == 2: #Beam Elements
                
                _, _, _, element_material, element_surface, _ = element
                
                K, M = self.KMBeam3D(x1,y1,z1,x2,y2,z2,
                                     element_material, element_surface)
                
                LG = np.array([node1.dof_id[0], node1.dof_id[1], node1.dof_id[2], node1.dof_id[3], node1.dof_id[4], node1.dof_id[5],
                               node2.dof_id[0], node2.dof_id[1], node2.dof_id[2], node2.dof_id[3], node2.dof_id[4], node2.dof_id[5]])
                LG = LG - 1
                num_of_dofs = 12
            
            # Now construct the full matrices 
            self.stiffnessFullMatrix, self.massFullMatrix = self.createFullMatrices(LG, K, M, num_of_dofs)
            
        # BUild the force vector 
        self.forceVector = self.createForceVector()
            
        self.K, self.M, self.F = self.removeDofs()
            
        self.K = 0.5 * (self.K + self.K.T) # Average for avoiding numerical errors
        # Calculate static displacement 
        K_inv = np.linalg.inv(self.K)
        self.staticDisplacement = K_inv @ self.F
                   
    
    def KMBar3D(self, x1,y1,z1,x2,y2,z2,
                            element_material, element_surface):
        
        material = element_material
                
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        cx = (x2 - x1) / L
        cy = (y2 - y1) / L
        cz = (z2 - z1) / L

        T = np.array([[cx, cy, cz, 0, 0, 0],
                    [0, 0, 0, cx, cy, cz]])

        KLocal = (material.elasticModulus * element_surface/ L) * np.array([[1, -1],
                                                     [-1, 1]])
    
        K = T.T @ KLocal @ T  # @ is the matrix multiplication operator in Python

        MLocal = (1/6) * material.density * element_surface * L * np.array([[2, 0, 0, 1, 0, 0],
                                                                [0, 2, 0, 0, 1, 0],
                                                                [0, 0, 2, 0, 0, 1],
                                                                [1, 0, 0, 2, 0, 0],
                                                                [0, 1, 0, 0, 2, 0],
                                                                [0, 0, 1, 0, 0, 2]])

        M = MLocal

        return K, M

    
    def KMBeam3D(self, x1, y1, z1, x2, y2, z2, 
                 element_material, element_surface):
    
    
        material = element_material
        radius = (element_surface / 3.14159) ** 0.5
        
        J = (3.14159 * radius**4) / 4
        I = (3.14159 * radius**4) / 4
        
        is_reduced = False
        
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        k = 9/10
        C = np.diag([material.elasticModulus * element_surface, k * material.shearModulus * element_surface, 
                     k * material.shearModulus * element_surface,
                    material.shearModulus * J, material.elasticModulus * I, 
                    material.elasticModulus * I])

        # Calculate K
        K = np.zeros((12, 12))
        if is_reduced:
            xi = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
            w = [1, 1]
            for i, xi_val in enumerate(xi):
                _, B = self.GetNB(xi_val, L)
                detJ = 0.5 * L
                K += w[i] * detJ * B.T @ C @ B
        else:
            xi = 0
            w = 2
            _, B = self.GetNB(xi, L)
            detJ = 0.5 * L
            K += w * detJ * B.T @ C @ B

        T = self.GetTBeam(x1, y1, z1, x2, y2, z2)
        K = T.T @ K @ T

        # Calculate M
        A = np.diag([element_surface, element_surface, 
                     element_surface, J, I, I])
        
        M = np.zeros((12, 12))
        xi = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        w = [1, 1]
        i = 0
        #for i, xi_val in enumerate(xi):
        for xi_val in xi:
            N, _ = self.GetNB(xi_val, L)
            detJ = 0.5 * L
            M += w[i] * material.density * detJ * N.T @ A @ N
            i += 1

        return K, M

    
    def createFullMatrices(self, LG, K, M, num_of_dofs):
        MFull = self.massFullMatrix
        KFull = self.stiffnessFullMatrix
        
        for jj in range(num_of_dofs):
            for kk in range(num_of_dofs):
                
                KFull[LG[jj], LG[kk]] = KFull[LG[jj], LG[kk]] + K[jj, kk]
                MFull[LG[jj], LG[kk]] = MFull[LG[jj], LG[kk]] + M[jj, kk]
                
        return KFull, MFull
   
    
    def createForceVector(self):
        
        nodeMatrix = self.preprocessor.nodeMatrix
        Ffull = self.forceVector
        
        for node in nodeMatrix:
            
            if any(node.force):
                
                Ffull[node.dof_id[0] - 1] = node.force[0]
                Ffull[node.dof_id[1] - 1] = node.force[1]
                Ffull[node.dof_id[2] - 1] = node.force[2]
                Ffull[node.dof_id[3] - 1] = node.force[3]
                Ffull[node.dof_id[4] - 1] = node.force[4]
                Ffull[node.dof_id[5] - 1] = node.force[5]
    
    
        return Ffull
    
    
    def GetNB(self, xi, L):
        
        N1 = 0.5 * (1 - xi)
        N2 = 0.5 * (1 + xi)
        N11 = N1 * np.ones(6)
        N22 = N2 * np.ones(6)
        N = np.block([[np.diag(N11), np.diag(N22)]])

        N1x = -0.5 * 2 / L
        N2x = 0.5 * 2 / L
        N1xx = N1x * np.ones(6)
        N2xx = N2x * np.ones(6)
        B = np.block([[np.diag(N1xx), np.diag(N2xx)]])
    
        B[1, 5] = -N1  # Adjusted index for 0-based indexing
        B[1, 11] = -N2  # Adjusted index for 0-based indexing
        B[2, 4] = N1  # Adjusted index for 0-based indexing
        B[2, 10] = N2  # Adjusted index for 0-based indexing

        return N, B    
    

    def GetTBeam(self, x1, y1, z1, x2, y2, z2):
        
        T = np.zeros((12, 12))
        xa = np.array([x1, y1, z1])
        xb = np.array([x2, y2, z2])
        xc = np.array([1, 0, 0])
    
        # Compute directional vectors
        x1 = xb - xa
        x3 = np.cross(x1, xc)
        x2 = np.cross(x3, x1)
    
        # Normalize vectors
        x1 = x1 / np.linalg.norm(x1)
        x2 = x2 / np.linalg.norm(x2)
        x3 = x3 / np.linalg.norm(x3)
    
        # Create transformation matrix 't' from normalized vectors
        t = np.vstack([x1, x2, x3])  # vstack stacks arrays in sequence vertically (row wise)

        # Populate the transformation matrix 'T'
        T[0:3, 0:3] = t
        T[3:6, 3:6] = t
        T[6:9, 6:9] = t
        T[9:12, 9:12] = t

        return T
    
    
    def removeDofs(self):
        
        dofs_to_delete = np.array(self.preprocessor.dofsToDelete) - 1

        # Remove specified DOFs from K
        K = np.delete(self.stiffnessFullMatrix, dofs_to_delete, axis=0)  # Remove rows
        K = np.delete(K, dofs_to_delete, axis=1)  # Remove columns

        # Remove specified DOFs from M
        M = np.delete(self.massFullMatrix, dofs_to_delete, axis=0)  # Remove rows
        M = np.delete(M, dofs_to_delete, axis=1)  # Remove columns

        # Remove specified DOFs from F
        F = np.delete(self.forceVector, dofs_to_delete)

        return K, M, F
    
    
    def eigenAnalysis(self):
        eigenvalues, eigenvectors = eigh(self.K , self.M)
        
        eigenfrequencies = np.sqrt(eigenvalues)
        
        self.eigenModes = eigenvectors
        self.eigenfrequencies = eigenfrequencies
        
    
    def RayleighDamping(self):
        
        # Damping factors for 1st and 6th eigenmode
        z1 = 0.01
        z6 = 0.02 
        
        
        # Define the matrix for the linear system
        A = np.array([[1, self.eigenfrequencies[0]**2],  # eigs[0] is the first eigenvalue in Python (0-based indexing)
                    [1, self.eigenfrequencies[5]**2]])  # eigs[5] is the sixth eigenvalue

        # Define the right-hand side vector
        b = np.array([[2*z1*self.eigenfrequencies[0]], 
                    [2*z6*self.eigenfrequencies[5]]])

        # Solve for ab
        ab = np.linalg.solve(A, b)

        # Calculate C
        self.C = ab[0]*self.M + ab[1]*self.K

     
    def solve_with_Newmark(self):
        
        # Initilize parameters for Newmark
        a_newmark = 0.25
        b_newmark = 0.5
        n = self.K.shape[0] # Number of degrees of freedom
        h = self.timeStep # Time Step
        w = self.eigenfrequencies[1] # Angular Frequency
        tf = self.simulationTime # Final Time 
        
        t, x, dx, ddx = Newmark(self.M, self.C, self.K, self.F, w, h, tf, a_newmark, b_newmark, n)
        
        self.t_Newmark = t  # time
        self.x_Newmark = x # Displacement
        self.dx_Newmark = dx # Velocity
        self.ddx_Newmark = ddx # Acceleration
        
        self.strain_stress_calculation()
    
    
    def solve_with_eigenAnalysis(self):
        
    
    
        t, x = eigenAnalysis(self.M, self.C, self.K, self.F, self.eigenfrequencies, self.eigenModes, 
                      self.timeStep, self.simulationTime)
        
        self.t_eigenAnalysis = t
        self.x_eigenAnalysis = x
        
        self.strain_stress_calculation()
        self.static_strain_stress_calc()
     
        
    def frequencyResponse(self, max_freq = None,  dof_interest = 726, step = 0.5):
        """
        Generate the Amplitude - Frequency repsponse diagram for specific node
        This diagram represent the steady state for harmonic load : F = F0 * sin(Omega * t) where omega = second eigenfrequency
        """        
        if max_freq is None:
            max_freq = self.eigenfrequencies[5]
            
        W = np.arange(0, max_freq + step, step)  # Frequency range
        num_freqs = len(W)
        
         # Initialize the arrays for vertices
        Xvs = np.zeros(num_freqs)
        Yvs = np.zeros(num_freqs)
        Zvs = np.zeros(num_freqs)
    
        Fs = np.concatenate([np.zeros((self.F.shape[0], 1)), -self.F[:, np.newaxis]], axis=0)  # Assuming F is a column vector
    
        for i, omega in enumerate(W):
            matrix = np.block([
                [self.K - (omega**2) * self.M, omega * self.C],
                [omega * self.C, (omega**2) * self.M - self.K]
            ])
        
            v = np.linalg.solve(matrix, Fs)
        
            # Assuming the response vector v is split into cosine and sine parts
            
            v_c = v[:int(v.shape[0]/2)]  # Cosine part
            v_s = v[int(v.shape[0]/2):]  # Sine part
        
            R = np.sqrt(v_c**2 + v_s**2)
        
            # Extract the response for specific DOFs
            VS = R[dof_interest:dof_interest+3]  # Adjust indices for your DOFs of interest
        
            Xvs[i] = VS[0]
            Yvs[i] = VS[1]
            Zvs[i] = VS[2]
        
        return W, Xvs, Yvs, Zvs
    
    
    
    def strain_stress_calculation(self):
        
        
        if self.x_eigenAnalysis is not None: 
            displacements = self.x_eigenAnalysis

        elif self.x_Newmark is not None:
            displacements = self.x_Newmark
            
            
        dofsNumber = self.preprocessor.totalDofs
        totalElements =  self.preprocessor.totalElements
        n_steps = displacements.shape[1]
        structural_properties = np.empty((totalElements, n_steps), dtype=object)
        
        global_displacements = np.zeros([dofsNumber, n_steps])
        remainingDofs = np.setdiff1d(np.arange(dofsNumber), self.preprocessor.dofsToDelete - 1) 
        global_displacements[remainingDofs, :] = displacements

        
        for ii in range(n_steps):
            
            for index, element in enumerate(self.preprocessor.elementMatrix, start=0):
                node_1, node_2, kind, material, surface, undeformed_length = element
                elastic_modulus = material.elasticModulus


                
                if kind == 1:
                
                    disp_first_node = global_displacements[node_1.dof_id[0:3] - 1, ii]
                    disp_second_node = global_displacements[node_2.dof_id[0:3] - 1, ii]
                    
                    new_coords_1 = self.preprocessor.nodeMatrix[node_1.node_id - 1].coords + disp_first_node
                    new_coords_2 = self.preprocessor.nodeMatrix[node_2.node_id - 1].coords + disp_second_node
                    
                    diff = np.abs(new_coords_2 - new_coords_1)
                    
                    deforemed_length = np.sqrt(np.sum(diff**2))
                    deltaL = deforemed_length - undeformed_length
                    ex = deltaL/undeformed_length
                    sx = elastic_modulus * ex
                    
                    structural_properties[index, ii] = Structural_Properties(ex = ex, sx = sx)
                
                elif kind ==2:
                    
                    disp_first_node = global_displacements[node_1.dof_id - 1, ii]
                    disp_second_node = global_displacements[node_2.dof_id - 1, ii]

                    new_coords_1 = self.preprocessor.nodeMatrix[node_1.node_id - 1].coords + disp_first_node[0:3]
                    new_coords_2 = self.preprocessor.nodeMatrix[node_2.node_id - 1].coords + disp_second_node[0:3]
                    diff = np.abs(new_coords_2 - new_coords_1)
                    
                    deforemed_length = np.sqrt(np.sum(diff**2))
                    
                    deltaL = deforemed_length - undeformed_length
                                    
                    _, B = self.GetNB(0, deforemed_length)
                    
                    node_disp = np.vstack((disp_first_node, disp_second_node)).reshape(-1, 1)
                    deformations = B @ node_disp
                    stresses = elastic_modulus * deformations

                    
                    structural_properties[index, ii] = Structural_Properties(ex = deformations[0], ey = deformations[1], ez = deformations[2],
                                                                             sx = stresses[0], sy = stresses[1], sz = stresses[2])
                    
                    self.structural_properties = structural_properties

    def static_strain_stress_calc(self):
        
        
        displacements = self.staticDisplacement
            
            
        dofsNumber = self.preprocessor.totalDofs
        totalElements =  self.preprocessor.totalElements
        n_steps = 1
        structural_properties = np.empty((totalElements, n_steps), dtype=object)
        
        global_displacements = np.zeros([dofsNumber, 1])
        remainingDofs = np.setdiff1d(np.arange(dofsNumber), self.preprocessor.dofsToDelete - 1) 
        global_displacements[remainingDofs,0] = displacements

        
            
        for index, element in enumerate(self.preprocessor.elementMatrix, start=0):
            node_1, node_2, kind, material, surface, undeformed_length = element
            elastic_modulus = material.elasticModulus


                
            if kind == 1:
                
                disp_first_node = global_displacements[node_1.dof_id[0:3] - 1]
                disp_second_node = global_displacements[node_2.dof_id[0:3] - 1]
                    
                new_coords_1 = self.preprocessor.nodeMatrix[node_1.node_id - 1].coords + disp_first_node
                new_coords_2 = self.preprocessor.nodeMatrix[node_2.node_id - 1].coords + disp_second_node
                    
                diff = np.abs(new_coords_2 - new_coords_1)
                    
                deforemed_length = np.sqrt(np.sum(diff**2))
                deltaL = deforemed_length - undeformed_length
                ex = deltaL/undeformed_length
                sx = elastic_modulus * ex
                    
                structural_properties[index] = Structural_Properties(ex = ex, sx = sx)
                
            elif kind ==2:
                    
                disp_first_node = global_displacements[node_1.dof_id - 1]
                disp_second_node = global_displacements[node_2.dof_id - 1]

                new_coords_1 = self.preprocessor.nodeMatrix[node_1.node_id - 1].coords + disp_first_node[0:3]
                new_coords_2 = self.preprocessor.nodeMatrix[node_2.node_id - 1].coords + disp_second_node[0:3]
                diff = np.abs(new_coords_2 - new_coords_1)
                    
                deforemed_length = np.sqrt(np.sum(diff**2))
                    
                deltaL = deforemed_length - undeformed_length
                                    
                _, B = self.GetNB(0, deforemed_length)
                    
                node_disp = np.vstack((disp_first_node, disp_second_node)).reshape(-1, 1)
                deformations = B @ node_disp
                stresses = elastic_modulus * deformations

                    
                structural_properties[index] = Structural_Properties(ex = deformations[0], ey = deformations[1], ez = deformations[2],
                                                                        sx = stresses[0], sy = stresses[1], sz = stresses[2])
                    
                self.static_structural_properties = structural_properties       