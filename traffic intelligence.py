import pandas as pd
import numpy as np from sklearn.model_selection
import train_test_split from sklearn.preprocessing 
import StandardScaler from sklearn.ensemble 
import Random ForestRegressor from sklearn.metrics
import mean_absolute_error, mean_squared_error 
import tensorflow as tf from tensorflow.keras.models 
import Sequential from tensorflow.keras.layers import Dense, Dropout 
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)

data = pd.DataFrame({

'time_of_day': np.random.randint(0, 24, 1000),

'day_of_week':

np.random.randint(0, 7, 1000),

'weather_condition':

np.random.choice(['sunny', 'rainy', 'cloudy'], 1000),
'road_type':

np.random.choice(['highway', 'urban', 'rural'], 1000),

'traffic_volume':

np.random.normal(0, 50, 1000) })

# Preprocess data
X = pd.get_dummies(data.dr op('traffic_volume', axis=1), columns=['weather_condition', 'road_type'], drop_first=True) y = data['traffic_volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models

rf_model = Random ForestR egressor(n_estimators=100, random state=42)

# Train models
rf_model = Random ForestR

egressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

nn_model = Sequential([Dense(128, activation='relu', input_shape=(X_train.shape[1],)), Dropout(0.2), Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])

nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate models

y_pred_rf = rf_model.predict(X_test)

y_pred_nn = nn_model.predict(X_test).flatten()

print("Random Forest Model:")

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("Neural Network Model:") print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_nn)) print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_nn)))

# Plot actual vs predicted values 
plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred_rf, alpha=0.5) 
plt.xlabel("Actual Traffic Volume") 
plt.ylabel("Predicted Traffic Volume") 
plt.title("Actual vs Predicted Traffic Volume (Random Forest)")
plt.show()

plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred_nn, alpha=0.5) 
plt.xlabel("Actual Traffic Volume") 
plt.ylabel("Predicted Traffic Volume") 
plt.title("Actual vs Predicted Traffic Volume (Neural Network)")
plt.show()
