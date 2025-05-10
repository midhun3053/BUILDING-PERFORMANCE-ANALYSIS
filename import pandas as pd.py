import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Simulate IoT Data (Energy + Comfort)
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.normal(24, 2, 100),
    'humidity': np.random.normal(50, 10, 100),
    'co2': np.random.normal(600, 100, 100),
    'occupancy': np.random.randint(0, 2, 100),
    'daylight': np.random.normal(200, 50, 100),
    'energy_consumption': np.random.normal(150, 30, 100)  # Target 1
})
# Derived target: comfort index (simplified)
data['comfort_index'] = 100 - (abs(data['temperature'] - 23) + abs(data['humidity'] - 45))

# Step 2: Prepare Data
features = ['temperature', 'humidity', 'co2', 'occupancy', 'daylight']
X = data[features]
y_energy = data['energy_consumption']
y_comfort = data['comfort_index']

# Step 3: Train AI Models
X_train, X_test, y_train_energy, y_test_energy = train_test_split(X, y_energy, test_size=0.2, random_state=0)
_, _, y_train_comfort, y_test_comfort = train_test_split(X, y_comfort, test_size=0.2, random_state=0)

model_energy = RandomForestRegressor(n_estimators=100)
model_comfort = RandomForestRegressor(n_estimators=100)

model_energy.fit(X_train, y_train_energy)
model_comfort.fit(X_train, y_train_comfort)

# Step 4: Predict
pred_energy = model_energy.predict(X_test)
pred_comfort = model_comfort.predict(X_test)

# Step 5: Evaluation
rmse_energy = mean_squared_error(y_test_energy, pred_energy, squared=False)
rmse_comfort = mean_squared_error(y_test_comfort, pred_comfort, squared=False)

print(f"Energy Prediction RMSE: {rmse_energy:.2f}")
print(f"Comfort Prediction RMSE: {rmse_comfort:.2f}")

# Step 6: Visualization
plt.figure(figsize=(12, 5))

# Energy plot
plt.subplot(1, 2, 1)
plt.plot(y_test_energy.values, label="Actual Energy")
plt.plot(pred_energy, label="Predicted Energy")
plt.title("Energy Consumption Prediction")
plt.xlabel("Test Sample")
plt.ylabel("Energy (kWh)")
plt.legend()

# Comfort plot
plt.subplot(1, 2, 2)
plt.plot(y_test_comfort.values, label="Actual Comfort Index", color='green')
plt.plot(pred_comfort, label="Predicted Comfort Index", color='orange')
plt.title("Comfort Index Prediction")
plt.xlabel("Test Sample")
plt.ylabel("Comfort Index")
plt.legend()

plt.tight_layout()
plt.show()
