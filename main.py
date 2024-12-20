from rdkit import Chem
import pandas as pd

# Load the QM9 dataset
qm9_data = pd.read_csv('qm9.csv')


# Function to check for simple alcohols while excluding ethers
def is_simple_alcohol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Check that the molecule contains only C, H, and O atoms
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {'C', 'H', 'O'}:
            return False

    # Check that all carbons are sp³ hybridized
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetHybridization() != Chem.HybridizationType.SP3:
            return False

    # Exclude ethers (R-O-R) using SMARTS pattern
    ether_pattern = Chem.MolFromSmarts('[CX4][O][CX4]')  # Ethers have two sp³ carbons attached to an oxygen
    if mol.HasSubstructMatch(ether_pattern):
        return False

    # Check for alcohol group (-OH) attached to sp³ carbon
    alcohol_pattern = Chem.MolFromSmarts('[CX4][OH]')  # Alcohols have an -OH group attached to an sp³ carbon
    return mol.HasSubstructMatch(alcohol_pattern)



qm9_data['is_simple_alcohol'] = qm9_data['smiles'].apply(is_simple_alcohol)

# Filter the dataset to keep only simple alcohols
simple_alcohols = qm9_data[qm9_data['is_simple_alcohol'] == True]

# Count total number of alcohols found (excluding ethers)
total_alcohols = simple_alcohols.shape[0]


simple_alcohols.to_csv('qm9_simple_alcohols_excluding_ethers.csv', index=False)


print(f"Filtered {total_alcohols} simple alcohols (excluding ethers) from the QM9 dataset.")
