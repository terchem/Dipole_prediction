from rdkit import Chem
import pandas as pd

qm9_data = pd.read_csv('qm9.csv')

def is_simple_alcohol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {'C', 'H', 'O'}:
            return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetHybridization() != Chem.HybridizationType.SP3:
            return False
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[CX4][O][CX4]')):
        return False
    return mol.HasSubstructMatch(Chem.MolFromSmarts('[CX4][OH]'))

qm9_data['is_simple_alcohol'] = qm9_data['smiles'].apply(is_simple_alcohol)

simple_alcohols = qm9_data[qm9_data['is_simple_alcohol']][['smiles', 'mu']]
simple_alcohols.to_csv('simple_alcohols.csv', index=False)

print(len(simple_alcohols))
