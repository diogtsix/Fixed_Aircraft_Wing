
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '60'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import json

from solvers.structural_dynamics.preprocessor import Preprocessor
from solvers.structural_dynamics.solver import Solver

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from statsmodels.tsa.arima.model import ARIMA

from kan import *
import torch

class Comparison():
    
    def __init__(self, preprocessor = Preprocessor(), number_of_samples = 300):
        
        self.preprocessor = preprocessor
        self.solver = Solver(preprocessor = self.preprocessor)
        
        self.number_of_samples = number_of_samples
        self.LSTM = None
        self.LSTM_scaler = None
        self.ARIMA = None
        self.ARIMA_scaler = None
        self.KAN = None
        
    def create_dataset(self, number_of_samples=5, force_range=(100, 1000), file_path='datasets/displacement_time_series_data.xlsx'):
        # Define the force values range and initialize lists to collect data
        forces = np.linspace(force_range[0], force_range[1], number_of_samples)
        dataset = {
            'Force': [],
            'Displacement': [],
            'Time': []
        }

        for force in forces:
            # Create a new preprocessor instance with the updated force value
            preprocessor = Preprocessor(forceValue=force)
            
            # Create a new solver instance using this preprocessor
            solver = Solver(preprocessor=preprocessor)
            solver.solve_with_eigenAnalysis()
            
            # Retrieve displacement and time data
            x = solver.x_eigenAnalysis[726, :]
            t = solver.t_eigenAnalysis
            
            # Append the data to the dataset lists
            dataset['Force'].append(force)
            dataset['Displacement'].append(x.tolist())  # Convert to list for better storage
            dataset['Time'].append(t.tolist())  # Convert to list for better storage
        
        # Convert the lists to a DataFrame
        df = pd.DataFrame({
            'Force': dataset['Force'],
            'Displacement': [json.dumps(d) for d in dataset['Displacement']],  # Convert lists to JSON strings
            'Time': [json.dumps(t) for t in dataset['Time']]  # Convert lists to JSON strings
        })
        
        # Save the DataFrame to an Excel file
        df.to_excel(file_path, index=False)
        
        return df
    
    def get_dataFrame(self, file_path):
        df = pd.read_excel(file_path)
        df['Displacement'] = df['Displacement'].apply(json.loads)
        df['Time'] = df['Time'].apply(json.loads)
        
        return df
     
    def train_models(self):
        
        file_path = 'datasets/displacement_time_series_data.xlsx'
        
        if not os.path.exists(file_path):
            
            df = comparison.create_dataset(file_path=file_path, number_of_samples=self.number_of_samples)

        df =  self.get_dataFrame(file_path) # Convert Dataframe into appropriet format
        
        
        df['Displacement'] = df['Displacement'].apply(lambda x: np.array(x).flatten())
        df['Time'] = df['Time'].apply(lambda x: np.array(x).flatten())
    
        X_train, X_test, y_train, y_test, scaler = self.preprocess_data(df, test_size=0.2)
        
        self.LSTM = self.LSTM_model(X_train, y_train, X_test, y_test, epochs=150, batch_size=32)
        self.LSTM_scaler = scaler 
        
        self.ARIMA = self.ARIMA_model(y_train, y_test, order=(5, 1, 0))
        
        self.KAN = self.Kolmogorov_Arnold_Networks_model(X_train, y_train, X_test, y_test, epochs=10)

    
    def LSTM_model(self, X_train, y_train, X_test, y_test, epochs=150, batch_size=32):
        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        
        return model
    
    def Kolmogorov_Arnold_Networks_model(self, X_train, y_train, X_test, y_test, epochs=10):
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        # y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        # y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        dataset = {
            'train_input': X_train_tensor,
            'train_label': y_train_tensor,
            'test_input': X_test_tensor,
            'test_label': y_test_tensor
        }        
        
        
           # Adjust width and grid parameters as needed
        input_dim = X_train_tensor.shape[1]
        output_dim = y_train_tensor.shape[1]
        model = KAN(width=[X_train.shape[1], 10, 4,y_train.shape[1]], grid=3, k=3, seed=0, device="cpu")
    
        
        _ = model.train(dataset, opt="LBFGS", steps=epochs, lamb=0.01, lamb_entropy=10, lr=0.01,device="cpu",
                loss_fn=torch.nn.MSELoss())
        
        return model
    
    def ARIMA_model(self, y_train, y_test, order=(5, 1, 0)):
        # Since ARIMA is univariate, we need to fit it on each individual time series
        
        # Flatten the training data for ARIMA
        train_series = y_train.flatten()
        test_series = y_test.flatten()

        # Fit the ARIMA model
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()

        # Make predictions
        start_index = len(train_series)
        end_index = start_index + len(test_series) - 1
        predictions = model_fit.predict(start=start_index, end=end_index)

        # Compute evaluation metrics
        mse = mean_squared_error(test_series, predictions)
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mse)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        return model_fit
    
    def preprocess_data(self, df, sequence_length=100, test_size=0.2):
        # Shuffling
        df = df.sample(frac=1).reset_index(drop=True)

        # Extracting features and labels
        X = []
        y = []
        
        for index, row in df.iterrows():
            force = row['Force']
            displacements = np.array(row['Displacement'])
            
            half_length = len(displacements) // 2
            
            input_sequence = np.concatenate(([force], displacements[:half_length]))
            output_sequence = displacements[half_length:]

            X.append(input_sequence)
            y.append(output_sequence)

        X = np.array(X)
        y = np.array(y)
        
        # Normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Splitting into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test, scaler
    
    def plot_models(self, LSTM, ARIMA, forceValue):
        
        
        preprocessor = Preprocessor(forceValue=forceValue)
        
        solver = Solver(preprocessor=preprocessor)
        solver.solve_with_eigenAnalysis()
        
        eigenSolution = solver.x_eigenAnalysis[726, :]
        t = solver.t_eigenAnalysis
        
              # Split the eigenSolution into two halves
        half_length = len(eigenSolution) // 2
        first_half = eigenSolution[:half_length]
        second_half = eigenSolution[half_length:]
        
        # Prepare input for the models
        input_sequence = np.concatenate(([forceValue], first_half))  # Include the force value
        
        input_sequence_normalized = self.LSTM_scaler.transform([input_sequence])  # Assuming scaler is stored
        
        # Reshape for LSTM prediction
        input_sequence_reshaped = input_sequence_normalized.reshape((1, len(input_sequence_normalized[0]), 1))
        
        # Predict the next half using LSTM
        lstm_prediction = LSTM.predict(input_sequence_reshaped)[0]
        
        # Predict the next half using ARIMA
        start_index = len(first_half)
        end_index = start_index + len(second_half) - 1
        arima_prediction = ARIMA.predict(start=start_index, end=end_index)
        
        # Predict the next half using KAN
        input_sequence_tensor = torch.tensor(input_sequence_normalized, dtype=torch.float32)
        kan_prediction = self.KAN(input_sequence_tensor).detach().numpy()[0]
    
    
        # Calculate Mean Absolute Squared Error (MASE) for LSTM and ARIMA
        mase_lstm = mean_absolute_error(second_half, lstm_prediction)
        mase_arima = mean_absolute_error(second_half, arima_prediction)
        mase_kan = mean_absolute_error(second_half, kan_prediction)

        # Plot the results
        plt.figure(figsize=(14, 8))
        
        # Plot the original time series
        plt.plot(eigenSolution, label='Original Time Series (EigenAnalysis)', color='blue', linestyle='solid')
        
        # Plot the ARIMA predictions
        arima_full_series = np.concatenate((first_half, arima_prediction))
        plt.plot(range(half_length, len(eigenSolution)), arima_full_series[half_length:], label=f'ARIMA Predictions (MASE: {mase_arima:.4f})', color='red', linestyle='dashed')
        
        # Plot the LSTM predictions
        lstm_full_series = np.concatenate((first_half, lstm_prediction))
        plt.plot(range(half_length), first_half, label='LSTM Input (First Half)', color='green', linestyle='solid')
        plt.plot(range(half_length, len(eigenSolution)), lstm_full_series[half_length:], label=f'LSTM Predictions (MASE: {mase_lstm:.4f})', color='green', linestyle='dashed')
        
        # Plot the KAN predictions
        kan_full_series = np.concatenate((first_half, kan_prediction))
        plt.plot(range(half_length, len(eigenSolution)), kan_full_series[half_length:], label=f'KAN Predictions (MASE: {mase_kan:.4f})', color='purple', linestyle='dashed')
        
    
        # Add title and labels
        plt.title(f'Time Series Predictions for Force Value: 500')
        plt.xlabel('Time Steps')
        plt.ylabel('Displacement')
        plt.legend()
        
        # Show plot
        plt.show()
        
        
        
# Example usage
if __name__ == "__main__":
    comparison = Comparison()
    comparison.train_models()
    
    comparison.plot_models(comparison.LSTM, comparison.ARIMA, 500)


