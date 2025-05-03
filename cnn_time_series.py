import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Step 1: Generate synthetic hourly temperature data
dates = pd.date_range(start='1/1/2020', periods=1000, freq='H')
temperature = np.sin(np.linspace(0, 50, 1000)) * 20 + 25  # Sinusoidal temperature pattern
df = pd.DataFrame({'Temperature': temperature}, index=dates)

# Step 2: Preprocess time series using sliding window
def preprocess_data(df, look_back=24):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:(i+look_back), 0].values)
        y.append(df.iloc[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 24
X, y = preprocess_data(df, look_back)

# Step 3: Reshape for CNN input: (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 4: Split into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Build CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 6: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Step 7: Predict and evaluate
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

train_rmse = np.sqrt(np.mean((trainPredict[:, 0] - y_train) ** 2))
test_rmse = np.sqrt(np.mean((testPredict[:, 0] - y_test) ** 2))
print(f'Train RMSE: {train_rmse:.3f}')
print(f'Test RMSE: {test_rmse:.3f}')

# Step 8: Plot results with proper labels
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(testPredict, label='Predicted Temperature')
plt.title('CNN Time Series Forecasting')
plt.xlabel('Time Step (hours)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
