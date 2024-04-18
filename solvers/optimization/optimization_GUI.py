import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QRadioButton, QGroupBox, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

from solvers.optimization.weight_optimization import Weight_Optimization

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Structural Optimization")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.init_parameters_ui()
        self.init_results_table()

    def init_parameters_ui(self):
        self.parameters_layout = QVBoxLayout()
        self.main_layout.addLayout(self.parameters_layout, 1)

        # Discrete values for element surfaces
        self.discrete_surface_values = QLineEdit("[0.0001, 0.0008, 0.0031, 0.008, 0.015]")
        self.parameters_layout.addWidget(QLabel("Discrete Surface Values"))
        self.parameters_layout.addWidget(self.discrete_surface_values)

        # Dataset size for training the Neural Network
        self.numberOfSamples = QLineEdit("30000")
        self.parameters_layout.addWidget(QLabel("Number of Samples in the Dataset for Neural Network"))
        self.parameters_layout.addWidget(self.numberOfSamples)
        
        # Number of Iterations for the GA optimizer
        self.GA_max_iter = QLineEdit("50")
        self.parameters_layout.addWidget(QLabel("Max Iterations for Genetic Algorithm Optimization"))
        self.parameters_layout.addWidget(self.GA_max_iter)
        
        # Population Size for Genetic algorithm opitmizer
        self.GA_population_size = QLineEdit("100")
        self.parameters_layout.addWidget(QLabel("Population Size for Genetic Algorithm Optimization"))
        self.parameters_layout.addWidget(self.GA_population_size)
        
        self.submit_button = QPushButton("Run Optimization")
        self.submit_button.clicked.connect(self.run_simulation)
        self.parameters_layout.addWidget(self.submit_button)

        self.setup_choice_buttons()

    def setup_choice_buttons(self):
        self.choice_group_box = QGroupBox("Optimization Method")
        self.choice_layout = QVBoxLayout()
        self.genetic_algorithm_opt = QRadioButton("Genetic Algorithm Optimization")
        self.ML_model_optimization = QRadioButton("ML Model with Genetic Algorithm Optimization")
        self.genetic_algorithm_opt.setChecked(True)
        self.choice_layout.addWidget(self.genetic_algorithm_opt)
        self.choice_layout.addWidget(self.ML_model_optimization)
        self.choice_group_box.setLayout(self.choice_layout)
        self.parameters_layout.addWidget(self.choice_group_box)

    def init_results_table(self):
        self.results_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_layout.addWidget(self.results_table)
        self.main_layout.addLayout(self.results_layout, 1)

    def run_simulation(self):
        
        surface_values_text = self.discrete_surface_values.text()
        numberOfSamples_text = self.numberOfSamples.text()
        GA_max_iter_text = self.GA_max_iter.text()
        GA_population_size_text = self.GA_population_size.text()
        try:
            surface_values = [float(val.strip()) for val in surface_values_text.strip('[]').split(',')]
            numberOfSamples = int(numberOfSamples_text)
            GA_max_iter = int(GA_max_iter_text)
            GA_population_size = int(GA_population_size_text)
            
            opt_with_ml = self.ML_model_optimization.isChecked()
            opt = Weight_Optimization(optimize_with_surrogate_ML_model=opt_with_ml, 
                                      discrete_surface_values=surface_values, 
                                      numberOfSamples=numberOfSamples, 
                                      GA_max_iter = GA_max_iter, 
                                      GA_population_size = GA_population_size)
            opt.run_optimization()
            self.update_results_table(opt.results)
            print("Optimization completed successfully.")
        except ValueError as e:
            print("Error parsing surface values:", e)
        except Exception as e:
            print("Error during Model Building:", e)
            
            

    def update_results_table(self, data):
        # Optionally add a header before adding new data
        header_description = "Results for: " + ("ML Model Optimization" if data.get("Optimization Method") == "Surrogated Neural Network Optimization" else "Objective Function Optimization")
        self.add_results_header(header_description)

        # Get the current number of rows
        current_rows = self.results_table.rowCount()

        # Increase the row count to accommodate new data
        self.results_table.setRowCount(current_rows + len(data))

        # Add new data at the end of the table
        for i, (key, value) in enumerate(data.items(), start=current_rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(key))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
            
    def add_results_header(self, header_description):
        # Get the current number of rows
        current_rows = self.results_table.rowCount()
        
        # Increase the row count for the header
        self.results_table.setRowCount(current_rows + 1)
        
        # Create a header row with merged cells
        header_item = QTableWidgetItem(header_description)
        header_item.setFlags(Qt.ItemIsEnabled)  # Make the header not editable
        header_item.setBackground(Qt.lightGray)  # Set a distinct background

        # Span the header across all columns
        self.results_table.setItem(current_rows, 0, header_item)
        self.results_table.setSpan(current_rows, 0, 1, self.results_table.columnCount())
        
                 
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()