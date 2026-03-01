import requests

COMMON_MOLECULES = {
    'aspirin':      'CC(=O)Oc1ccccc1C(=O)O',
    'caffeine':     'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'ibuprofen':    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'paracetamol':  'CC(=O)Nc1ccc(O)cc1',
    'dopamine':     'NCCc1ccc(O)c(O)c1',
    'serotonin':    'NCCc1c[nH]c2ccc(O)cc12',
    'morphine':     'OC1=CC=C2CC3N(C)CCC34C2=C1OC4',
    'penicillin':   'CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1',
    'glucose':      'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'ethanol':      'CCO',
    'benzene':      'c1ccccc1',
}

def name_to_smiles(name):
    """Convert molecule name to SMILES via PubChem API."""
    if name.lower() in COMMON_MOLECULES:
        return COMMON_MOLECULES[name.lower()]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()['PropertyTable']['Properties'][0]
            return data.get('IsomericSMILES') or data.get('SMILES')
    except:
        pass
    return None