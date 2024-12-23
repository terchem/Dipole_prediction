from rdkit import Chem
from rdkit.Chem import Draw


smiles = "CC(O)CCC(C)(C)O"  # Ethanol


mol = Chem.MolFromSmiles(smiles)
# Check if the molecule object is valid
if mol:
    
    img = Draw.MolToImage(mol)
    img.show()  # Opens the image in the default viewer
else:
    print("Invalid SMILES string")
