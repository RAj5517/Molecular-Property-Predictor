import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fingerprint(smiles):
    """Convert SMILES string to Morgan fingerprint array."""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprintAsNumPy(mol)

def smiles_to_descriptors(smiles):
    """Compute physicochemical descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW':       round(Descriptors.MolWt(mol), 2),
        'LogP':     round(Descriptors.MolLogP(mol), 2),
        'HBD':      Descriptors.NumHDonors(mol),
        'HBA':      Descriptors.NumHAcceptors(mol),
        'TPSA':     round(Descriptors.TPSA(mol), 2),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
    }

def lipinski_check(descriptors):
    """Check Lipinski Rule of Five."""
    rules = {
        'MW ≤ 500':          descriptors['MW'] <= 500,
        'LogP ≤ 5':          descriptors['LogP'] <= 5,
        'HB Donors ≤ 5':     descriptors['HBD'] <= 5,
        'HB Acceptors ≤ 10': descriptors['HBA'] <= 10,
    }
    passed = sum(rules.values())
    return rules, passed