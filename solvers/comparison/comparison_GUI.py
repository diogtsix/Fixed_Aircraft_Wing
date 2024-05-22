import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QGroupBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from solvers.comparison.comparison_main import Comparison

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Comparison")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.init_parameters_ui()
        self.init_plot_ui()
        
        self.setLayoutStretch()

    def init_parameters_ui(self):
        self.parameters_container = QGroupBox("Parameters")
        self.parameters_layout = QVBoxLayout()

        self.force_value, self.force_value_input = self.create_input_field("Force Value:", "500")
        self.epochs, self.epochs_input = self.create_input_field("LSTM Epochs:", "150")
        self.iterations, self.iterations_input = self.create_input_field("KAN Iterations:", "5")
        self.grid_size, self.grid_size_input = self.create_input_field("KAN Grid Size:", "17")
        self.order_k, self.order_k_input = self.create_input_field("KAN Order k:", "7")

        self.parameters_layout.addWidget(self.force_value)
        self.parameters_layout.addWidget(self.epochs)
        self.parameters_layout.addWidget(self.iterations)
        self.parameters_layout.addWidget(self.grid_size)
        self.parameters_layout.addWidget(self.order_k)

        self.run_button = QPushButton("Run Comparison")
        self.run_button.clicked.connect(self.run_comparison)
        self.parameters_layout.addWidget(self.run_button)

        self.parameters_container.setLayout(self.parameters_layout)
        self.parameters_container.setFixedWidth(300)
        self.main_layout.addWidget(self.parameters_container)

    def init_plot_ui(self):
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        self.plot_container.setLayout(self.plot_layout)
        self.main_layout.addWidget(self.plot_container)

    def create_input_field(self, label_text, default_value=""):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setText(default_value)
        input_field.setObjectName(label_text.strip(':').replace(' ', '_'))
        layout.addWidget(label)
        layout.addWidget(input_field)
        widget = QWidget()
        widget.setLayout(layout)
        return widget, input_field

    def run_comparison(self):
        force_value = float(self.force_value_input.text())
        lstm_epochs = int(self.epochs_input.text())
        kan_iterations = int(self.iterations_input.text())
        kan_grid_size = int(self.grid_size_input.text())
        kan_order_k = int(self.order_k_input.text())

        comparison = Comparison()
        comparison.LSTM_epochs = lstm_epochs
        comparison.KAN_steps = kan_iterations
        comparison.KAN_grid = kan_grid_size
        comparison.KAN_k = kan_order_k

        comparison.train_models()
        comparison.plot_models(comparison.LSTM, comparison.ARIMA, force_value)

        self.update_plot(comparison.comparison_figure)

    def update_plot(self, fig):
        self.canvas.figure.clf()
        self.canvas.figure = fig
        self.canvas.draw()
        
    def setLayoutStretch(self):
        self.main_layout.setStretch(0, 1)
        self.main_layout.setStretch(1, 3)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
