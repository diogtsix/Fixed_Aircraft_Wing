import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Postprocess():
    
    def __init__(self, solver, ax_handle = None ):
        
        
        self.solver = solver
        self.preprocessor = solver.preprocessor
        self.ax = ax_handle
        self.animation = None
        

    def plotEigenModes(self, numberOfModes, scaling_factor = 5):
        
        cols_per_row = 3
        rows = 2
        modes_per_figure = rows * cols_per_row
    
        total_figures = (numberOfModes + modes_per_figure - 1) // modes_per_figure
    
        for fig_idx in range(total_figures):
            start_mode_idx = fig_idx * modes_per_figure
            end_mode_idx = min(start_mode_idx + modes_per_figure, numberOfModes)
            modes_in_this_figure = end_mode_idx - start_mode_idx
        
            rows_in_this_figure = (modes_in_this_figure + cols_per_row - 1) // cols_per_row
            if modes_in_this_figure <= cols_per_row:
                rows_in_this_figure = 1  # If all modes fit in one row
        
            fig, axs = plt.subplots(rows_in_this_figure, cols_per_row, figsize=(18, rows_in_this_figure * 6), subplot_kw={'projection': '3d'})
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])  # Make it an array if it's not (for a single subplot)
            axs = axs.ravel()  # Flatten the array
        
            for mode_plot_idx in range(modes_in_this_figure):
                mode_idx = start_mode_idx + mode_plot_idx
                m = self.solver.eigenModes[:, mode_idx] * scaling_factor
                v_global = self.create_v_global(mode=m)
            
                ax = axs[mode_plot_idx]
            
                
                self.plot_3D_structure(ax=ax, v_global=v_global)
                eigenfrequency = self.solver.eigenfrequencies[mode_idx]
                ax.set_title(f'$\omega_{{{mode_idx + 1}}} = {eigenfrequency:.2f}$ Hz')
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_zlabel('Z [m]')
                ax.grid(True)
                ax.set_xlim([-0.75, 1.3])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([0, 4])
                
            plt.tight_layout()
            plt.show()
    
    
    def plot_3D_structure(self, ax, v_global , undeformed_color='k'):
        
        wingElementMatrix = self.preprocessor.elementMatrix
        
        # First, plot the undeformed structure
        for element in wingElementMatrix:
            start_node, end_node, _ = element
            x_values_undeformed = [start_node.coords[0], end_node.coords[0]]
            y_values_undeformed = [start_node.coords[1], end_node.coords[1]]
            z_values_undeformed = [start_node.coords[2], end_node.coords[2]]
        
            # Plot the undeformed structure in a neutral color
            ax.plot(x_values_undeformed, y_values_undeformed, z_values_undeformed, color=undeformed_color, marker='o', markersize=3, linestyle='--', linewidth=1)
    
    
        for element in wingElementMatrix:
            start_node, end_node, kind = element
            
            if kind == 1:
                color = 'b'
            elif kind == 2:
                color = 'r'
                
            x_values = [start_node.coords[0] + v_global[start_node.dof_id[0] - 1],
                        end_node.coords[0] + v_global[end_node.dof_id[0] - 1]]
            y_values = [start_node.coords[1] + v_global[start_node.dof_id[1] - 1],
                        end_node.coords[1] + v_global[end_node.dof_id[1] - 1]]
            z_values = [start_node.coords[2] + v_global[start_node.dof_id[2] - 1],
                        end_node.coords[2] + v_global[end_node.dof_id[2] - 1]]
            ax.plot(x_values, y_values, z_values, color=color, marker='o', markersize=5, linestyle='-', linewidth=1.5)
         
            
    def create_v_global(self, mode, scaling_factor=1):

        dofsNumber = self.preprocessor.totalDofs
        v_global = np.zeros(dofsNumber)
        
        remainingDofs = np.setdiff1d(np.arange(dofsNumber), self.preprocessor.dofsToDelete - 1)  # Adjusting indices, assuming dofsToDelete is 1-based
        
        # Assuming mode vector length matches the number of remaining DOFs after deletion
        # This part may need adjustment based on how you define modes and handle indexing
        v_global[remainingDofs] = mode * scaling_factor
        
        return v_global
    
    
    def simulation_displacements(self, scaling_factor = 1):
        
        # Ensure eigenanalysis has been solved
        #self.solver.solve_with_eigenAnalysis()
        if self.solver.x_eigenAnalysis is not None: 
            
            displacements = self.solver.x_eigenAnalysis
            timeSteps = self.solver.t_eigenAnalysis

                      
        elif self.solver.x_Newmark is not None:
            
            displacements = self.solver.x_Newmark
            timeSteps = self.solver.t_Newmark  
            
        n_steps = displacements.shape[1]
    
        # Add Boundary condition dofs
        dofsNumber = self.preprocessor.totalDofs
        global_displacements = np.zeros([dofsNumber, n_steps])
    
        remainingDofs = np.setdiff1d(np.arange(dofsNumber), self.preprocessor.dofsToDelete - 1)  # Adjust if dofsToDelete is 1-based
    
        # Apply scaling factor to displacements and update global_displacements
        global_displacements[remainingDofs, :] = displacements * scaling_factor
    
        ani = self.visualize_structure(global_displacements, timeSteps)
        
        return ani
    
    def visualize_structure(self, global_displacements, timeSteps):
        
        
        if self.ax == None :
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            elevation_angle = 0
            azimuth_angle = 0 
            ax.view_init(elev=elevation_angle, azim=azimuth_angle)
        else:
            fig = self.ax.figure
            ax = fig.add_subplot(111, projection='3d')
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            
        ani = FuncAnimation(fig, self.update_plot, frames=range(timeSteps.shape[0]), 
                            fargs=(ax, global_displacements, 
                                   self.solver.preprocessor.nodeMatrix, self.solver.preprocessor.elementMatrix),
                            interval=100)
        

        if self.ax == None :
            plt.show()
        else:
            self.ax.draw()
            
        return ani
    
    def update_plot(self, step, ax, global_displacements, nodeMatrix, elementMatrix):
        ax.clear()

        # Iterate through each element and plot it using the updated positions
        for element in elementMatrix:
            start_node, end_node, kind = element
            start_dofs = self.extract_translational_dofs_for_node(global_displacements, start_node.node_id)[:, step]
            end_dofs = self.extract_translational_dofs_for_node(global_displacements, end_node.node_id)[:, step]
            
            start_pos = nodeMatrix[start_node.node_id - 1].coords + start_dofs
            end_pos = nodeMatrix[end_node.node_id - 1].coords + end_dofs
            
            if kind == 1:
                color = 'b-'
            elif kind == 2:
                color = 'r-'
                
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], color, marker='o', markersize=5, linewidth=1.5)
            
       # ax.set_title(f'Time: {timeSteps[step]:.2f} s')
  

        ax.set_title('Real Time Translational simulation for Harmonic Loading  F = F0 sin(Ωt) with Ω = 115.28 Hz')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-0.75, 1.3])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 4])
        
    
    def extract_translational_dofs_for_node(self, global_displacements, node_id):
        # Extracts x, y, z displacements for a specific node across all timesteps
        # Node IDs are assumed to start from 1, adjust indices accordingly
        # Each node has 6 DOFs, with the first 3 being translational (x, y, z)
        start_index = (node_id - 1) * 6  # Starting index for the node's DOFs
        return global_displacements[start_index:start_index + 3, :]

    def frequencyResponse(self, dofOfInterest = 726):
        
        W, Xvs, Yvs, Zvs = self.solver.frequencyResponse(step = 2, dof_interest = dofOfInterest)
        

    
        fig, axs = plt.subplots(1, 3, figsize=(18, 1 * 6))
        
        axs[0].plot(W, Xvs)
        axs[1].plot(W, Yvs)
        axs[2].plot(W, Zvs)
        
        axs[0].set_title('Frequency - Response Diagram')
        axs[0].set_xlabel('Omega [Hz]')
        axs[0].set_ylabel('X Displacement [m]')
        
        axs[1].set_title('Frequency - Response Diagram')
        axs[1].set_xlabel('Omega [Hz]')
        axs[1].set_ylabel('Y Displacement [m]')
        
        axs[2].set_title('Frequency - Response Diagram')
        axs[2].set_xlabel('Omega [Hz]')
        axs[2].set_ylabel('Z Displacement [m]')
        
        plt.show()
        
        