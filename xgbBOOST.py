import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the dataset
file_path = 'successful_conformers.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)

# Define the target variable (mu) and features (all other columns except target and non-relevant columns)
target = 'mu'
features = [col for col in data.columns if col != target and col != 'smiles' and col != 'ConformerSuccess']

# Drop rows with missing data
data_cleaned = data.dropna()

# Separate the features (X) and target (y)
X = data_cleaned[features]
y = data_cleaned[target]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting model using XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Root Mean Squared Error (RMSE) with XGBoost: {rmse}")

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label='Perfect Prediction')
plt.xlabel('Actual Dipole Moment')
plt.ylabel('Predicted Dipole Moment')
plt.title('Actual vs Predicted Dipole Moment (XGBoost)')
plt.legend()
plt.show()
