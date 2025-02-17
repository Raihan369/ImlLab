# -*- coding: utf-8 -*-
"""IML_Lstm Lab 2025.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YFRzG99iqwisAf8t-PBNGd1aFM4mKvAK
"""

!pip install tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('/content/AirPassengers.csv')
data = data[['#Passengers']]  # Use the correct column name

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 50
X, y = create_sequences(scaled_data, sequence_length)
X = np.expand_dims(X, axis=-1)  # Add feature dimension for LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Function to build the LSTM model with random layers and neurons
def build_model(units, learning_rate, optimizer, activation, dropout_rate, num_layers):
    model = Sequential([Input(shape=(sequence_length, 1))])  # Input layer
    for i in range(num_layers - 1):  # First layers return sequences
        model.add(LSTM(units, activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units, activation=activation, return_sequences=False))  # Last LSTM layer without return_sequences
    model.add(Dense(1))
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mse')
    return model

result = []

def run_model(activation, optimizer, num_layers, units, epochs, batch_size):
    model = build_model(units, 0.001, optimizer, activation, 0.02, num_layers)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    train_loss = history.history['loss'][-1]
    test_loss = history.history['val_loss'][-1]

    result.append({
        "Activation": activation,
        "Optimizer": optimizer.__name__,
        "Num Layers": num_layers,
        "Neurons": units,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Train Loss": train_loss,
        "Test Loss": test_loss
    })
    print(f"Activation: {activation}, Optimizer: {optimizer.__name__}, Num Layers: {num_layers}, Neurons: {units}, "
          f"Epochs: {epochs}, Batch Size: {batch_size}, Train Loss: {train_loss}, Test Loss: {test_loss}")

for i in range(10):
    activation = np.random.choice(['relu', 'tanh', 'sigmoid'])
    optimizer = np.random.choice([Adam, SGD])
    num_layers = int(np.random.choice([1, 2, 3]))  # Randomly choose the number of LSTM layers between 1 to 3
    units = int(np.random.choice([32, 64, 128, 256]))  # Force units to be an integer
    epochs = int(np.random.choice([20, 50, 100]))  # Randomly choose epochs
    batch_size = int(np.random.choice([16, 32, 64]))  # Randomly choose batch size

    run_model(activation, optimizer, num_layers, units, epochs, batch_size)

# Save the result to a CSV file
result_df = pd.DataFrame(result)
result_df.to_csv('/content/B190305008.csv', index=False)
print("Results saved to 'B190305008.csv'")