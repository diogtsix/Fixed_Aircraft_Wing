import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Postprocess():
    
    def __init__(self, solver):
        
        
        self.solver = solver
        self.preprocess = solver.preprocessor
        
        

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
        
        wingElementMatrix = self.preprocess.elementMatrix
        
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

        dofsNumber = self.preprocess.totalDofs
        v_global = np.zeros(dofsNumber)
        
        remainingDofs = np.setdiff1d(np.arange(dofsNumber), self.preprocess.dofsToDelete - 1)  # Adjusting indices, assuming dofsToDelete is 1-based
        
        # Assuming mode vector length matches the number of remaining DOFs after deletion
        # This part may need adjustment based on how you define modes and handle indexing
        v_global[remainingDofs] = mode * scaling_factor
        
        return v_global