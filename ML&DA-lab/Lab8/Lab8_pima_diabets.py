import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


df = pd.read_csv("diabetes.csv")
# Exemplu coloane: ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

X = df[['DiabetesPedigreeFunction']].values  # input feature
y = df[['Age']].values                        # target


# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)


# Reshape for LSTM (samples, timesteps, features)
# Folosim timesteps=1, pentru că avem doar o valoare de intrare
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# Train
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)


# Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # revenim la scara originală


# Evaluate
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(df['Age'].iloc[y_test.shape[0]*-1:], y_pred)
r2 = r2_score(df['Age'].iloc[y_test.shape[0]*-1:], y_pred)

print(f"MSE: {mse:.2f}, R^2: {r2:.2f}")
