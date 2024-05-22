import sys
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from solvers.structural_dynamics.StructuralDynamics_main import MainWindow as StructuralDynamics
from solvers.optimization.optimization_GUI import MainWindow as StructuralOptimization
from solvers.predictive_maintance.predictive_maintance_GUI import MainWindow as PredictiveMaintance
from solvers.comparison.comparison_GUI import MainWindow as Comparison

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose Fixed Wing Problem to Solve: ")

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        # layout = QVBoxLayout()
        layout = QVBoxLayout(central_widget)
        
        # Create buttons
        btn_structural_dynamics = QPushButton("Fixed Wing Structural Dynamics Analysis")
        btn_optimization = QPushButton("Fixed Wing Structural Optimization")
        btn_predictive_maintance = QPushButton("Fatigue Damage and Tolerance Analysis")
        btn_comparison = QPushButton("ML Models Comparison For Wings Response Prediction")


        # Connect buttons to functions
        btn_structural_dynamics.clicked.connect(self.run_structural_dynamics)
        btn_optimization.clicked.connect(self.run_optimization)
        btn_predictive_maintance.clicked.connect(self.run_predictive_maintance)
        btn_comparison.clicked.connect(self.run_ML_comparison)

        
        # Add buttons to layout
        layout.addWidget(btn_structural_dynamics)
        layout.addWidget(btn_optimization)
        layout.addWidget(btn_predictive_maintance)
        layout.addWidget(btn_comparison)
        


    def run_structural_dynamics(self):
    
        self.structural_dynamics_window = StructuralDynamics()
        self.structural_dynamics_window.show()

    def run_optimization(self):
        
        self.structural_optimization_window = StructuralOptimization()
        self.structural_optimization_window.show()

    def run_predictive_maintance(self):
        self.predictive_maintance_window = PredictiveMaintance()
        self.predictive_maintance_window.show()
        
        
    def run_ML_comparison(self):
        self.comparison_window = Comparison()
        self.comparison_window.show()
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()