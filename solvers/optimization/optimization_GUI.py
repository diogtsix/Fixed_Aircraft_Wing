import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QFormLayout, QStackedLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import traceback

from PyQt5.QtWidgets import QRadioButton, QGroupBox, QHBoxLayout

from solvers.optimization.weight_optimization import Weight_Optimization


class OptimizationThread(QThread):
    
    finished_signal = pyqtSignal(str)
    def __init__(self, opt_with_ml, surface_values):
        super().__init__()
        self.opt_with_ml = opt_with_ml
        self.surface_values = surface_values

    def run(self):
        try:
            opt = Weight_Optimization(optimize_with_surrogate_ML_model=self.opt_with_ml, 
                                      discrete_surface_values=self.surface_values)
            opt.run_optimization()
            self.finished_signal.emit("Optimization completed successfully.")  # Emit success message
        except Exception as e:
            error_message = f"Failed to run optimization: {str(e)}"
            print(error_message)
            traceback.print_exc()
            self.finished_signal.emit(error_message)  # Emit error message
            
            
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Structural Optimization")
        
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        
        # Left layout for parameters
        self.parameters_layout = QVBoxLayout()
        self.main_layout.addLayout(self.parameters_layout, 1)

        # Create input fields
        self.discrete_surface_values = QLineEdit()

        self.submit_button = QPushButton("Run Optimization")
        self.submit_button.clicked.connect(self.run_simulation)
        
        # Set default values
        self.discrete_surface_values.setText("[0.0001, 0.0008, 0.0031, 0.008, 0.015]")  # Default number of airfoils
        # Add inputs to parameters layout
        self.parameters_layout.addWidget(QLabel("Discrete Surface Values"))
        self.parameters_layout.addWidget(self.discrete_surface_values)
        self.parameters_layout.addWidget(self.submit_button)
        
        self.setup_choice_buttons()
    
    def run_simulation(self):
        
        surface_values_text   = self.discrete_surface_values.text()  
        
        try:
            
            surface_values = [float(val.strip()) for val in surface_values_text.strip('[]').split(',')]
            opt_with_ml = self.ML_model_optimization.isChecked()
            self.thread = OptimizationThread(opt_with_ml, surface_values)  # Store as an instance variable.
            self.thread.finished_signal.connect(self.optimization_finished)
            self.thread.start()
            
            # if self.genetic_algorithm_opt.isChecked():
                
            #     opt = Weight_Optimization(optimize_with_surrogate_ML_model= False, 
            #                 discrete_surface_values = surface_values)

            #     opt.run_optimization()
                
            # elif self.ML_model_optimization.isChecked():

            #     opt = Weight_Optimization(optimize_with_surrogate_ML_model= True, 
            #                 discrete_surface_values = surface_values)

            #     opt.run_optimization()
                
        except ValueError as e:
            print("Error parsing surface values:", e)
        except Exception as e:
            print("Error during Model Building:", e)
                    
            
    def setup_choice_buttons(self):
        
        
        # Group Box to hold the radio buttons
        self.choice_group_box = QGroupBox("Results to Visualize")
        self.choice_layout = QVBoxLayout()
    
        # Radio buttons for choices
        self.genetic_algorithm_opt = QRadioButton("Genetic Algorithm Optimization")
        self.ML_model_optimization = QRadioButton("ML Model with Genetic Algorithm Optimization")
    
        # Set default selection
        self.genetic_algorithm_opt.setChecked(True)

        # Add buttons to the layout
        self.choice_layout.addWidget(self.genetic_algorithm_opt)
        self.choice_layout.addWidget(self.ML_model_optimization)
    
        # Add the layout to the group box
        self.choice_group_box.setLayout(self.choice_layout)

        # Add the group box to the main parameters layout
        self.parameters_layout.addWidget(self.choice_group_box)
        
    def optimization_finished(self, message):
        print(message)  # Log or show the message in the GUI.   
        
def main():
    
    app =  QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
        