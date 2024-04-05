import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from solvers.structural_dynamics.StructuralDynamics_main import MainWindow as StructuralDynamics


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Choose Fixed Wing Problem to Solve: ")

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout()

        # Create buttons
        btn_structural_dynamics = QPushButton("Fixed Wing Structural Dynamics Analysis")
        btn_optimization = QPushButton("Fixed Wing Structural Optimization")

        # Connect buttons to functions
        btn_structural_dynamics.clicked.connect(self.run_structural_dynamics)
        btn_optimization.clicked.connect(self.run_optimization)

        # Add buttons to layout
        layout.addWidget(btn_structural_dynamics)
        layout.addWidget(btn_optimization)

        # Set layout to central widget
        central_widget.setLayout(layout)

    def run_structural_dynamics(self):
    
        self.structural_dynamics_window = StructuralDynamics()
        self.structural_dynamics_window.show()

    def run_optimization(self):
        pass

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()