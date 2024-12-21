import pandas as pd


data = {
    "smiles": ["C", "CC", "CO", "C(CO)O", "CCO", "O=C=O", "CC(=O)O", "C1=CC=CC=C1"],
    "mu": [0.00, 0.05, 1.69, 0.0075, 1.70, 0.00, 1.75, 0.00],  # Dipole moments in Debye
    "molecular_weight": [16.04, 30.07, 32.04, 62.07, 46.07, 44.01, 60.05, 78.11],  # Molecular weights in g/mol
    "num_atoms": [1, 2, 3, 5, 4, 3, 4, 6],  # Number of atoms in the molecule
}

# Convert to pandas DataFrame
qm9_data = pd.DataFrame(data)
new_data= qm9_data[["smiles","num_atoms"]]
# Display the DataFrame
print(new_data)
