from rdkit import Chem
from rdkit.Chem import Draw

# Define a SMILES string
smiles = "C(CCO)CCO"  # Ethanol

# Convert SMILES to an RDKit molecule object
mol = Chem.MolFromSmiles(smiles)
# Check if the molecule object is valid
if mol:
    # Render and save the molecule as an image
    img = Draw.MolToImage(mol)
    img.show()  # Opens the image in the default viewer
else:
    print("Invalid SMILES string")