
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ascunde WARNINGS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # permite duplicate LLVM/OpenMP

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv('/home/tabita17/Documents/SCHOOL/M1/IAADPI/iad_doc/ML&DA-lab/Lab8/monthly_milk_production.csv', index_col='Date', parse_dates=True)
df.index.freq = 'MS'

# Plot producția
df.plot(figsize=(12,6), title='Monthly Milk Production')
plt.show()

# Split date train/test
train = df.iloc[:150]  # primele 150 de luni ...12 ani (168 in totoal)
test = df.iloc[150:]   # restul pentru test

#Scale data
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# TimeseriesGenerator
n_input = 12  # folosim 12 luni pentru previziune
n_features = 1

train_generator = TimeseriesGenerator(scaled_train, scaled_train,
                                      length=n_input, batch_size=1)

# Definim LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
epochs = 10
model.fit(train_generator, epochs=epochs, verbose=1)

#  Predict
# Generăm secvențe pentru test
n_input_test = min(12, len(scaled_test))
test_generator = TimeseriesGenerator(scaled_test, scaled_test,
                                     length=n_input_test, batch_size=1)

predictions_scaled = model.predict(test_generator)
predictions = scaler.inverse_transform(predictions_scaled).flatten()

#  Actual test values (deplasate pentru a se potrivi cu n_input) ---
actual = test[n_input:].values.flatten()  # eliminăm primele n_input valori

# Compute metrics
mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(12,6))
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Milk Production: Actual vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Production')
plt.legend()
plt.show()
