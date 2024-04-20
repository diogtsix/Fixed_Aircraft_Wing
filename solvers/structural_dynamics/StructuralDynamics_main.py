import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QFormLayout, QStackedLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QRadioButton, QGroupBox, QHBoxLayout
from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver
from solvers.structural_dynamics.postprocess import Postprocess



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Structural Dynamics Analysis")
        self.animation = None 
        
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        
        # Left layout for parameters
        self.parameters_layout = QVBoxLayout()
        self.main_layout.addLayout(self.parameters_layout, 1)

        # Right layout for matplotlib canvas
        self.figure_layout = QVBoxLayout()
        self.main_layout.addLayout(self.figure_layout, 3)
        
        # Create input fields
        self.num_of_airfoils_input = QLineEdit()
        self.force_input = QLineEdit()
        self.timestep_input = QLineEdit()
        self.simulaton_time_input = QLineEdit()
        self.num_of_eigenmodes_input = QLineEdit()
        self.scaling_factor_input = QLineEdit()
        self.node_of_interest = QLineEdit()
        
        self.submit_button = QPushButton("Run Simulation")
        self.submit_button.clicked.connect(self.run_simulation)
        
                # Set default values
        self.num_of_airfoils_input.setText("11")  # Default number of airfoils
        self.force_input.setText("500")  # Default force value
        self.timestep_input.setText("0.05")  # Default timestep value
        
        self.simulaton_time_input.setText("1")  # Default number of airfoils
        self.num_of_eigenmodes_input.setText("6")  # Default force value
        self.scaling_factor_input.setText("5")  # Default timestep value
        self.node_of_interest.setText("100")  # Default timestep value

        # Add inputs to parameters layout
        self.parameters_layout.addWidget(QLabel("Number of Airfoils:"))
        self.parameters_layout.addWidget(self.num_of_airfoils_input)
        self.parameters_layout.addWidget(QLabel("Force [N]:"))
        self.parameters_layout.addWidget(self.force_input)
        self.parameters_layout.addWidget(QLabel("Timestep:"))
        self.parameters_layout.addWidget(self.timestep_input)
        self.parameters_layout.addWidget(QLabel("Simulation Time [sec]:"))
        self.parameters_layout.addWidget(self.simulaton_time_input)
        self.parameters_layout.addWidget(QLabel("Num of EigenModes to Visualize:"))
        self.parameters_layout.addWidget(self.num_of_eigenmodes_input)        
        self.parameters_layout.addWidget(QLabel("Scaling Factor for Visualization:"))
        self.parameters_layout.addWidget(self.scaling_factor_input)  
        self.parameters_layout.addWidget(QLabel("Node of Interest:"))
        self.parameters_layout.addWidget(self.node_of_interest)        
        
        self.parameters_layout.addWidget(self.submit_button)
        
        self.setup_choice_buttons_solver()
        self.setup_choice_buttons_results()
        
        
        # Setup button for stress visualization option
        self.setup_visualize_stresses_button()

        
        # Add the matplotlib FigureCanvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.figure_layout.addWidget(self.canvas)

    def run_simulation(self):
        # Placeholder for running the simulation
        # Extract values from input fields and run your simulations here
        num_of_airfoils = int(self.num_of_airfoils_input.text())
        force = float(self.force_input.text())
        timestep = float(self.timestep_input.text())
        simTime = float(self.simulaton_time_input.text()) 
        numOfEigenmodes = int(self.num_of_eigenmodes_input.text()) 
        scalingFactor =  float(self.scaling_factor_input.text())
        nodeOfInterest =  int(self.node_of_interest.text())

        pre, solve = self.solve_model(num_of_airfoils, force, scalingFactor,timestep, 
                         simTime)

        try:
        # ... existing code ...
        
            if self.eigenmode_button.isChecked():
                # Run EigenMode plot simulation
                
                self.eigenmode_plots( scalingFactor, numOfEigenmodes, 
                         solve)
                
            elif self.frequency_response_button.isChecked():
                # Run Frequency Response simulation
                
                self.frequency_response_plots(nodeOfInterest, solve )
                
            elif self.real_time_simulation_button.isChecked():
                # Run Real Time simulation
                
                visualize_stresses = self.stress_visualization_button.isChecked()
                
                self.real_time_sim(scalingFactor, solve, visualize_stresses)
                
                
        except Exception as e:
            print("Error during simulation or plotting:", e)



    def real_time_sim(self, scalingFactor, solve , visualize_stresses):
        

        post = Postprocess(solver=solve, ax_handle = self.canvas, visualize_stresses = visualize_stresses)
        self.animation = post.simulation_displacements(scaling_factor= scalingFactor)
      
        #self.canvas.draw()
    
    
    
    def frequency_response_plots(self,  dof_of_interest, 
                                 solve ):
        
        post = Postprocess(solver=solve)
        post.frequencyResponse(dofOfInterest= dof_of_interest)

    
    def eigenmode_plots(self, scalingFactor, numOfEigenmodes, 
                         solve):
        

      
        post = Postprocess(solver=solve)
        post.plotEigenModes(numberOfModes = numOfEigenmodes, scaling_factor = scalingFactor)

        
    def solve_model(self, num_of_airfoils, force, scalingFactor,timestep, 
                            simTime):
        
        # Create preprocessors object
        pre = Preprocessor(numberOfAirfoils= num_of_airfoils, forceValue= force)
        
        # Solve Wing       
        solve = Solver(preprocessor= pre, timeStep= timestep, simulationTime= simTime)
        
        try:
            
            if self.eigenAnalysis_button.isChecked():
                solve.solve_with_eigenAnalysis()
            elif self.newmark_button.isChecked():
                solve.solve_with_Newmark()
                
        except Exception as e:
            print("Error during simulation:", e)        
        
        return pre, solve
        
    
    def setup_choice_buttons_results(self):
        
        
        # Group Box to hold the radio buttons
        self.choice_group_box = QGroupBox("Results to Visualize")
        self.choice_layout = QVBoxLayout()
    
        # Radio buttons for choices
        self.eigenmode_button = QRadioButton("EigenMode Plots")
        self.frequency_response_button = QRadioButton("Frequency Response")
        self.real_time_simulation_button = QRadioButton("Real Time Simulation")
    
        # Set default selection
        self.eigenmode_button.setChecked(True)

        # Add buttons to the layout
        self.choice_layout.addWidget(self.eigenmode_button)
        self.choice_layout.addWidget(self.frequency_response_button)
        self.choice_layout.addWidget(self.real_time_simulation_button)
    
        # Add the layout to the group box
        self.choice_group_box.setLayout(self.choice_layout)

        # Add the group box to the main parameters layout
        self.parameters_layout.addWidget(self.choice_group_box)

    def setup_choice_buttons_solver(self):
        
        
        # Group Box to hold the radio buttons
        self.choice_group_box = QGroupBox("Solver to use")
        self.choice_layout = QVBoxLayout()
    
        # Radio buttons for choices
        self.newmark_button = QRadioButton("Newmark")
        self.eigenAnalysis_button = QRadioButton("Eigenmethod Analysis")
    
        # Set default selection
        self.eigenAnalysis_button.setChecked(True)

        # Add buttons to the layout
        self.choice_layout.addWidget(self.newmark_button)
        self.choice_layout.addWidget(self.eigenAnalysis_button)

    
        # Add the layout to the group box
        self.choice_group_box.setLayout(self.choice_layout)

        # Add the group box to the main parameters layout
        self.parameters_layout.addWidget(self.choice_group_box)    
        
        
        
    def setup_visualize_stresses_button(self):
        # Group Box to hold the radio button
        self.stress_visualize_group_box = QGroupBox("Visualize Stresses")
        self.stress_visualize_layout = QVBoxLayout()
        
        # Radio button for stress visualization
        self.stress_visualization_button = QRadioButton("Visualize Stresses During Real-Time Simulation")
        self.stress_visualization_button.setChecked(False)  # Default not checked

        # Add button to the layout
        self.stress_visualize_layout.addWidget(self.stress_visualization_button)

        # Add the layout to the group box
        self.stress_visualize_group_box.setLayout(self.stress_visualize_layout)

        # Add the group box to the main parameters layout
        self.parameters_layout.addWidget(self.stress_visualize_group_box)
    
    
def main():
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()