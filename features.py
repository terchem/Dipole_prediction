import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem

# Load the dataset
file_path = "simp_alc.csv"  # Update this path to your file location
data = pd.read_csv(file_path)

# Ensure the dataset has a SMILES column
if "smiles" not in data.columns:
    raise ValueError("The dataset must contain a 'SMILES' column for RDKit processing.")


# Define the descriptor calculation function with 3D conformers
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # Add explicit hydrogens
            mol = Chem.AddHs(mol)

            # Generate 3D conformer
            params = AllChem.ETKDGv3()  # Use ETKDGv3 for better geometry
            params.maxAttempts = 10
            params.randomSeed = 42  # Reproducibility
            result = AllChem.EmbedMolecule(mol, params)

            if result != 0:  # Check if embedding was successful
                raise ValueError("Conformer generation failed")

            # Optimize the conformer geometry
            AllChem.UFFOptimizeMolecule(mol)

            # Calculate Gasteiger charges
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]

            # Calculate descriptors
            return {
                # Basic properties
                "MolWt": Descriptors.MolWt(mol),
                "ExactMolWt": Descriptors.ExactMolWt(mol),
                "TPSA": Descriptors.TPSA(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumOHGroups": rdMolDescriptors.CalcNumHBD(mol),  # Hydroxyl group count

                # 3D geometry descriptors
                "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),

                # Topological indices
                "Chi0n": Descriptors.Chi0n(mol),
                "Chi1n": Descriptors.Chi1n(mol),
                "Kappa1": Descriptors.Kappa1(mol),
                "RingCount": Descriptors.RingCount(mol),

                # Polarity and charges
                "MaxAbsPartialCharge": max(abs(c) for c in charges),
                "MinAbsPartialCharge": min(abs(c) for c in charges),


                # Conformer success flag
                "ConformerSuccess": 1,
            }
        except Exception as e:
            print(f"Error processing molecule {smiles}: {e}")
            return {"ConformerSuccess": 0}  # Track failed conformer generation
    else:
        return {"ConformerSuccess": 0}


# Calculate descriptors for each molecule in the dataset
descriptors = data["smiles"].apply(calculate_descriptors)
descriptor_df = pd.DataFrame(descriptors.tolist())

# Combine the descriptors with the original dataset
result = pd.concat([data, descriptor_df], axis=1)

# Filter for molecules with successful 3D conformers
successful_conformers = result[result["ConformerSuccess"] == 1]
total_success = successful_conformers.shape[0]

print(f"Total successful 3D conformers: {total_success}")

# Save the filtered dataset for future use
successful_conformers_path = "successful_conformers.csv"
successful_conformers.to_csv(successful_conformers_path, index=False)

print(f"Filtered dataset with successful conformers saved as: {successful_conformers_path}")

# Compute correlations for molecules with successful conformers
numeric_result = successful_conformers.select_dtypes(include=["number"])  # Select numeric columns only

if "mu" in numeric_result.columns:  # Ensure 'mu' exists in the numeric dataframe
    correlations = numeric_result.corr()["mu"].sort_values(ascending=False)
    print("Correlations with Dipole Moment (successful conformers only):")
    print(correlations)

    # Plot correlation heatmap and save it
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_result.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Successful Conformers Only)")
    plt.savefig("correlation_heatmap_successful.png", dpi=300)  # Save heatmap as PNG
    plt.show()
