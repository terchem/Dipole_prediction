import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = 'successful_conformers.csv'  # Ensure this file path is correct
data = pd.read_csv(file_path)

# target variable (mu) and features
target = 'mu'
features = [col for col in data.columns if col not in [target, 'smiles', 'ConformerSuccess']]


X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# test set
y_pred = model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# outliers (residuals > 2 standard deviations from the mean)
std_residual = residuals.std()
mean_residual = residuals.mean()
outlier_mask = (residuals.abs() > 2 * std_residual)

# map the test set indices to the original dataset
outlier_indices = residuals[outlier_mask].index

outliers = data.loc[outlier_indices]

# Generate topo
outlier_mols = [Chem.MolFromSmiles(smiles) for smiles in outliers['smiles'] if smiles]
img = Draw.MolsToGridImage(outlier_mols, molsPerRow=3, subImgSize=(300, 300))


img_path = "outliers_image.png"
img.save(img_path)


img = plt.imread(img_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()


outliers_info = outliers[['smiles']].copy()
outliers_info['Predicted_mu'] = pd.Series(y_pred, index=y_test.index)[outlier_indices].values
outliers_info['Actual_mu'] = y_test[outlier_indices].values


# Save the outliers information to a CSV file
outliers_info.to_csv('outliers_with_cas.csv', index=False)

# Plot all points (Actual vs Predicted) with outliers highlighted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Normal Points", alpha=0.6)
plt.scatter(y_test[outlier_mask], y_pred[outlier_mask], color="red", label="Outliers", alpha=0.8)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label="Perfect Prediction")
plt.xlabel("Actual mu")
plt.ylabel("Predicted mu")
plt.title("Actual vs Predicted mu with Outliers Highlighted")
plt.legend()
plt.savefig("actual_vs_predicted_with_outliers.png")  # Save the plot
plt.show()

# Display the list of outliers
print(outliers_info)
