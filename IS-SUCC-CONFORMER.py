import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def check_3d_conformer(smiles):
    """
    Check if a 3D conformer can be successfully generated for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            # Add explicit hydrogens
            mol = Chem.AddHs(mol)

            # Attempt to generate a 3D conformer
            params = AllChem.ETKDGv3()  # Use ETKDGv3 for better geometry
            params.maxAttempts = 10
            params.randomSeed = 42  # Reproducibility
            result = AllChem.EmbedMolecule(mol, params)

            if result == 0:  # Successful embedding returns 0
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error processing molecule {smiles}: {e}")
            return 0
    else:
        return 0


def main():
    # Load the dataset
    file_path = "simp_alc.csv"  # Replace with your input file
    data = pd.read_csv(file_path)

    # Ensure the dataset has a SMILES column
    if "smiles" not in data.columns:
        raise ValueError("The dataset must contain a 'SMILES' column for RDKit processing.")


    print("Checking 3D conformer generation...")
    data["ConformerSuccess"] = data["smiles"].apply(check_3d_conformer)

    
    output_path = "conformer_check_results.csv"
    data.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total successful conformers: {data['ConformerSuccess'].sum()}")


if __name__ == "__main__":
    main()
