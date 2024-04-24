import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QRadioButton, QGroupBox, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from solvers.predictive_maintance.predictive_maintance import Predictive_maintance

import numpy as np

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FD&T Analysis")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.init_parameters_ui()
        self.init_results_ui()

    def init_parameters_ui(self):
        # Container for parameter inputs
        self.parameters_container = QGroupBox("Parameters")
        self.parameters_layout = QVBoxLayout()

        # Material properties inputs
        self.material_properties_container = QGroupBox("Material Properties")
        self.material_properties_layout = QVBoxLayout()
        
        self.elastic_modulus, self.elastic_modulus_input = self.create_input_field("Elastic Modulus:", "73100000000")
        self.poisson_ratio , self.poisson_ratio_input = self.create_input_field("Poisson Ratio:", "0.33")
        self.density, self.density_input = self.create_input_field("Material Density :", "2780")
        self.shearModulus, self.shearModulus_input = self.create_input_field("Shear Modulus :", "28000000000")
        self.C, self.C_input = self.create_input_field("Material C constant :" , "7.9927")
        self.m, self.m_input = self.create_input_field("Material m constant : ", "-0.235")
        self.material_Name, self.material_Name_input = self.create_input_field("Material Name :", "AA 7075")
        
        self.material_properties_layout.addWidget(self.elastic_modulus)
        self.material_properties_layout.addWidget(self.poisson_ratio)
        self.material_properties_layout.addWidget(self.density)
        self.material_properties_layout.addWidget(self.shearModulus)
        self.material_properties_layout.addWidget(self.C)
        self.material_properties_layout.addWidget(self.m)
        self.material_properties_layout.addWidget(self.material_Name)
        
        self.material_properties_container.setLayout(self.material_properties_layout)

        # Loading conditions inputs
        self.loading_conditions_container = QGroupBox("Loading Conditions")
        self.loading_conditions_layout = QVBoxLayout()
        
        # Example default values for loads and cycles
        default_loads = ["500", "1000", "2000", "3000", "4000", "5000", "6000"]
        default_cycles = ["3000", "7000", "4000", "1000", "5000", "1000" , "3000"]
    
        # Creating a grid-like layout with 6 rows and 2 columns for Load and Cycle inputs
        for i in range(7):
            row_layout = QHBoxLayout()  # Each row has its own horizontal layout
            load_widget, load_input = self.create_input_field(f"Load {i+1}:", default_loads[i])
            cycle_widget, cycle_input = self.create_input_field(f"Num of Cycles {i+1}:", default_cycles[i])
            row_layout.addWidget(load_widget)  # Add the QWidget part of the tuple
            row_layout.addWidget(cycle_widget)  # Add the QWidget part of the tuple
            self.loading_conditions_layout.addLayout(row_layout)
            self.loading_conditions_container.setLayout(self.loading_conditions_layout)

        # Add all to parameters layout
        self.parameters_layout.addWidget(self.material_properties_container)
        self.parameters_layout.addWidget(self.loading_conditions_container)
        self.run_button = QPushButton("Run Calculation")
        self.run_button.clicked.connect(self.run_calculation)
        self.parameters_layout.addWidget(self.run_button)

        self.parameters_container.setLayout(self.parameters_layout)
        self.main_layout.addWidget(self.parameters_container)

    def init_results_ui(self):
        # Container for results display
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        
        # Title for the results section
        title_label = QLabel("Fatigue Analysis Results")
        title_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(title_label)
    

        # Placeholder for the S-N plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.results_layout.addWidget(self.canvas)

        # Table for displaying results
        self.results_table = QTableWidget(7, 5)  # 6 rows for conditions, 2 columns for Load and Cycle
        self.results_table.setHorizontalHeaderLabels([
            'σm [MPa]',  # Mean stress
            'σa [MPa]',  # Alternating stress
            'log(Nf)',  # Log cycles to failure
            'Nf',  # Cycles to failure
            'Accumulated Damage'
        ])
        self.results_layout.addWidget(self.results_table)
        
        # Additional results displayed
        self.num_cycles_label = QLabel("1) Number of blocks to Failure = ")
        self.fatigue_life_label = QLabel("2) Fatigue Life hours = ")
        self.total_damage_label = QLabel("3) Total Accumulated Damage = ")
        self.crack_initiation = QLabel("4) Crack Initiation = ")
        
        self.results_layout.addWidget(self.num_cycles_label)
        self.results_layout.addWidget(self.fatigue_life_label)
        self.results_layout.addWidget(self.total_damage_label)
        self.results_layout.addWidget(self.crack_initiation)

        self.results_container.setLayout(self.results_layout)
        self.main_layout.addWidget(self.results_container)

    def update_plot(self, fig):
        # Remove the current figure's content
        self.canvas.figure.clf()
        # Add the new figure content
        self.canvas.figure = fig
        self.canvas.draw()
        
    def create_input_field(self, label_text, default_value=""):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setText(default_value)
        input_field.setObjectName(label_text.strip(':').replace(' ', '_'))  # Set object name for easy retrieval
        layout.addWidget(label)
        layout.addWidget(input_field)
        widget = QWidget()
        widget.setLayout(layout)
        return widget, input_field

    def run_calculation(self):

        elastic_modulus = float(self.elastic_modulus_input.text())
        poisson_ratio = float(self.poisson_ratio_input.text())
        density = float(self.density_input.text())
        shearModulus = float(self.shearModulus_input.text())
        C = float(self.C_input.text())
        m  = float(self.m_input.text())
        material_Name = self.material_Name_input.text()
        
        loads = []
        cycles = []
        for i in range(7):  # Assuming you have 7 rows of load/cycle inputs
            load_input_name = f"Load_{i+1}"
            cycle_input_name = f"Num_of_Cycles_{i+1}"

            load_input = self.loading_conditions_container.findChild(QLineEdit, load_input_name)
            cycle_input = self.loading_conditions_container.findChild(QLineEdit, cycle_input_name)

            loads.append(float(load_input.text()))
            cycles.append(int(cycle_input.text()))
        
        # COnvert arrays into numoy 
        loads = np.array(loads)
        cycles = np.array(cycles)
            
        pmObj = Predictive_maintance(elasticModuluts = elastic_modulus,
                                     poissonRation = poisson_ratio,
                                     density = density,
                                     shearModulus=shearModulus,
                                     C = C,
                                     m = (-1)*m, 
                                     loads = loads, 
                                     cycles = cycles, 
                                     material_Name = material_Name)
        
        fig = pmObj.create_S_N_plots()
        self.update_plot(fig)
        
        if pmObj.crack_condition == False:
            crack = "No crack Initiation"
        else:
            crack = "Crack Initiated"
            
        self.update_additional_results(pmObj.number_of_blocks_to_failure,
                                       pmObj.fatigue_safe_life_hours, 
                                       pmObj.accumulated_damage_total,
                                       crack)
    
        self.update_results_table(pmObj.results)
        
    def update_additional_results(self, num_cycles, fatigue_life_hours, total_damage, crack):
        self.num_cycles_label.setText(f"1) Number of blocks to Failure = {num_cycles}")
        self.fatigue_life_label.setText(f"2) Fatigue Life hours = {fatigue_life_hours}")
        self.total_damage_label.setText(f"3) Total Accumulated Damage = {total_damage}")
        self.crack_initiation.setText(f"4) Crack Initiation = " + crack)
    
    def update_results_table(self, results):
        self.results_table.setRowCount(len(results))  # Set the number of rows based on results
        for row_index, row_data in enumerate(results):
            for col_index, item in enumerate(row_data):
               # Format the item using scientific notation
                cell_item = QTableWidgetItem(f"{item:.2e}")
                self.results_table.setItem(row_index, col_index, cell_item)
                           
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())