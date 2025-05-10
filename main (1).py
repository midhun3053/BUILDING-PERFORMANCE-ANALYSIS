import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# Simulate IoT data with unique smart building features
np.random.seed(2024)
data = pd.DataFrame({
    'temperature': np.random.normal(22, 2.5, 400),
    'humidity': np.random.normal(47, 11, 400),
    'co2': np.random.normal(610, 90, 400),
    'occupancy': np.random.randint(0, 4, 400),  # 0=empty,1=low,2=medium,3=high
    'daylight': np.random.normal(190, 55, 400),
    'hvac_power': np.random.normal(48, 14, 400),
    'window_open': np.random.randint(0, 2, 400),
    'smart_blinds_position': np.random.uniform(0, 1, 400),  # 0=closed, 1=open
    'air_purifier_status': np.random.randint(0, 2, 400),    # 0=off, 1=on
    'solar_panel_output': np.random.normal(20, 5, 400),     # kW generated onsite
    'energy_consumption': np.random.normal(155, 32, 400)
})
# Enhanced comfort index including air quality and smart blinds effect
data['comfort_index'] = 100 - (
    abs(data['temperature'] - 22) +
    abs(data['humidity'] - 48) +
    data['co2'] / 120 -
    5 * data['smart_blinds_position'] +
    3 * data['air_purifier_status']
)

# Features and targets
features = [
    'temperature', 'humidity', 'co2', 'occupancy', 'daylight',
    'hvac_power', 'window_open', 'smart_blinds_position',
    'air_purifier_status', 'solar_panel_output'
]
X = data[features]
y_energy = data['energy_consumption']
y_comfort = data['comfort_index']

# Train/test split
X_train, X_test, y_train_energy, y_test_energy = train_test_split(X, y_energy, test_size=0.25, random_state=2024)
_, _, y_train_comfort, y_test_comfort = train_test_split(X, y_comfort, test_size=0.25, random_state=2024)

# Train HistGradientBoosting models (fast and handles categorical-like features)
model_energy = HistGradientBoostingRegressor(max_iter=200, random_state=2024)
model_comfort = HistGradientBoostingRegressor(max_iter=200, random_state=2024)
model_energy.fit(X_train, y_train_energy)
model_comfort.fit(X_train, y_train_comfort)

# Predict
pred_energy = model_energy.predict(X_test)
pred_comfort = model_comfort.predict(X_test)

# Evaluate with MAE
mae_energy = mean_absolute_error(y_test_energy, pred_energy)
mae_comfort = mean_absolute_error(y_test_comfort, pred_comfort)

print(f"Energy Prediction MAE: {mae_energy:.2f}")
print(f"Comfort Prediction MAE: {mae_comfort:.2f}")

# Permutation feature importance for interpretability
perm_imp_energy = permutation_importance(model_energy, X_test, y_test_energy, n_repeats=15, random_state=2024)
perm_imp_comfort = permutation_importance(model_comfort, X_test, y_test_comfort, n_repeats=15, random_state=2024)

feat_imp_energy = pd.Series(perm_imp_energy.importances_mean, index=features).sort_values(ascending=False)
feat_imp_comfort = pd.Series(perm_imp_comfort.importances_mean, index=features).sort_values(ascending=False)

print("\nEnergy Model Permutation Importances:")
print(feat_imp_energy)
print("\nComfort Model Permutation Importances:")
print(feat_imp_comfort)

# Visualization
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.scatter(y_test_energy, pred_energy, alpha=0.6, color='navy')
plt.plot([y_test_energy.min(), y_test_energy.max()], [y_test_energy.min(), y_test_energy.max()], 'r--')
plt.title("Energy Consumption: Actual vs Predicted")
plt.xlabel("Actual Energy (kWh)")
plt.ylabel("Predicted Energy (kWh)")

plt.subplot(2, 3, 2)
feat_imp_energy.plot(kind='bar', color='mediumblue')
plt.title("Energy Model Feature Importance (Permutation)")
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
plt.barh(['Smart Blinds Position', 'Air Purifier Status', 'Solar Panel Output'],
         [data['smart_blinds_position'].mean(), data['air_purifier_status'].mean(), data['solar_panel_output'].mean()],
         color=['orange', 'green', 'gold'])
plt.title("Average Smart Tech Usage in Dataset")

plt.subplot(2, 3, 4)
plt.scatter(y_test_comfort, pred_comfort, alpha=0.6, color='darkgreen')
plt.plot([y_test_comfort.min(), y_test_comfort.max()], [y_test_comfort.min(), y_test_comfort.max()], 'r--')
plt.title("Comfort Index: Actual vs Predicted")
plt.xlabel("Actual Comfort")
plt.ylabel("Predicted Comfort")

plt.subplot(2, 3, 5)
feat_imp_comfort.plot(kind='bar', color='darkgreen')
plt.title("Comfort Model Feature Importance (Permutation)")
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
plt.boxplot([data['smart_blinds_position'], data['air_purifier_status'], data['solar_panel_output']],
            labels=['Smart Blinds', 'Air Purifier', 'Solar Output'])
plt.title("Smart Tech Feature Distributions")

plt.tight_layout()
plt.show()