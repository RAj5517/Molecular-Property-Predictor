import streamlit as st
import torch
import pickle
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

COMMON_MOLECULES = {
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'caffeine': 'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'acetaminophen': 'CC(=O)Nc1ccc(O)cc1',
    'dopamine': 'NCCc1ccc(O)c(O)c1',
    'serotonin': 'NCCc1c[nH]c2ccc(O)cc12',
    'morphine': 'OC1=CC=C2CC3N(C)CCC34C2=C1OC4',
    'penicillin': 'CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1',
    'glucose': 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'ethanol': 'CCO',
    'benzene': 'c1ccccc1',
}

def name_to_smiles(name):
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

# ── Page config ──
st.set_page_config(
    page_title="Molecular Property Predictor",
    page_icon="🧪",
    layout="wide"
)

HF_REPO = "RAj5517/mol-property-predictor"

# ── Multi-task model definition ──
class MultiTaskChemBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        hidden = self.backbone.config.hidden_size
        self.bbbp_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.esol_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.bbbp_head(cls), self.esol_head(cls)


# ── Load all models (cached) ──
@st.cache_resource
def load_models():
    # Random Forest
    rf_bbbp_path = hf_hub_download(repo_id=HF_REPO, filename="rf_bbbp.pkl")
    rf_esol_path = hf_hub_download(repo_id=HF_REPO, filename="rf_esol.pkl")
    with open(rf_bbbp_path, 'rb') as f:
        rf_bbbp = pickle.load(f)
    with open(rf_esol_path, 'rb') as f:
        rf_esol = pickle.load(f)

    # ChemBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Multi-task model
    mt_path = hf_hub_download(repo_id=HF_REPO, filename="multitask_v3_best.pt")
    mt_model = MultiTaskChemBERTa()
    mt_model.load_state_dict(torch.load(mt_path, map_location='cpu'))
    mt_model.eval()

    return rf_bbbp, rf_esol, tokenizer, mt_model



def get_mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300))
    return img


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW':        round(Descriptors.MolWt(mol), 2),
        'LogP':      round(Descriptors.MolLogP(mol), 2),
        'HBD':       Descriptors.NumHDonors(mol),
        'HBA':       Descriptors.NumHAcceptors(mol),
        'TPSA':      round(Descriptors.TPSA(mol), 2),
        'RotBonds':  Descriptors.NumRotatableBonds(mol),
    }


def lipinski_check(desc):
    rules = {
        'MW ≤ 500':         desc['MW'] <= 500,
        'LogP ≤ 5':         desc['LogP'] <= 5,
        'HB Donors ≤ 5':    desc['HBD'] <= 5,
        'HB Acceptors ≤ 10': desc['HBA'] <= 10,
    }
    passed = sum(rules.values())
    return rules, passed


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprintAsNumPy(mol).reshape(1, -1)


def predict_rf(smiles, rf_bbbp, rf_esol):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None, None
    bbbp_prob = rf_bbbp.predict_proba(fp)[0][1]
    esol_pred = rf_esol.predict(fp)[0]
    return bbbp_prob, esol_pred


def predict_multitask(smiles, tokenizer, mt_model):
    enc = tokenizer(
        smiles, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        bbbp_logit, esol_pred = mt_model(enc['input_ids'], enc['attention_mask'])
    bbbp_prob = torch.sigmoid(bbbp_logit).item()
    esol_val  = esol_pred.item()
    return bbbp_prob, esol_val


# ── UI ──
st.title("🧪 Molecular Property Predictor")
st.markdown("Predict **Blood-Brain Barrier penetration** and **Solubility** for any molecule.")
st.markdown("---")

# input
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_input(
        "Enter molecule name or SMILES",
        placeholder="e.g. Aspirin   or   CC(=O)Oc1ccccc1C(=O)O"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn and user_input:

    # resolve SMILES
    smiles = user_input.strip()
    if Chem.MolFromSmiles(smiles) is None:
        with st.spinner("Looking up molecule..."):
            smiles = name_to_smiles(user_input.strip())

    if smiles is None:
        st.error("Could not find molecule. Try entering SMILES directly.")
        st.stop()

    st.success(f"SMILES: `{smiles}`")

    # load models
    with st.spinner("Loading models..."):
        rf_bbbp, rf_esol, tokenizer, mt_model = load_models()

    # get descriptors and image
    desc = get_descriptors(smiles)
    img  = get_mol_image(smiles)

    # predictions
    with st.spinner("Running predictions..."):
        rf_bbbp_prob, rf_esol_pred     = predict_rf(smiles, rf_bbbp, rf_esol)
        mt_bbbp_prob, mt_esol_pred     = predict_multitask(smiles, tokenizer, mt_model)

    st.markdown("---")

    # layout
    left, right = st.columns([1, 2])

    with left:
        st.subheader("🔬 Molecule")
        if img:
            st.image(img, width=280)
        st.caption(f"Input: {user_input}")

    with right:
        st.subheader("📊 Predictions")

        # BBB results
        tab1, tab2 = st.tabs(["🧠 BBB Penetration", "💧 Solubility"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Random Forest",
                    f"{rf_bbbp_prob:.1%}",
                    "Crosses BBB" if rf_bbbp_prob > 0.5 else "Does NOT cross"
                )
            with c2:
                st.metric(
                    "Multi-task ChemBERTa",
                    f"{mt_bbbp_prob:.1%}",
                    "Crosses BBB" if mt_bbbp_prob > 0.5 else "Does NOT cross"
                )

            # consensus
            avg = (rf_bbbp_prob + mt_bbbp_prob) / 2
            if avg > 0.7:
                st.success(f"✅ Consensus: CROSSES BBB ({avg:.1%} confidence)")
            elif avg > 0.4:
                st.warning(f"⚠️ Consensus: UNCERTAIN ({avg:.1%} confidence)")
            else:
                st.error(f"❌ Consensus: DOES NOT cross BBB ({avg:.1%} confidence)")

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Random Forest",
                    f"{rf_esol_pred:.2f}",
                    "log mol/L"
                )
            with c2:
                st.metric(
                    "Multi-task ChemBERTa",
                    f"{mt_esol_pred:.2f}",
                    "log mol/L"
                )

            # interpret solubility
            avg_esol = (rf_esol_pred + mt_esol_pred) / 2
            if avg_esol > -1:
                st.success("💧 Highly soluble")
            elif avg_esol > -3:
                st.info("💧 Moderately soluble")
            elif avg_esol > -5:
                st.warning("💧 Poorly soluble")
            else:
                st.error("💧 Very poorly soluble")

    st.markdown("---")

    # Lipinski + descriptors
    st.subheader("🧬 Molecular Descriptors")
    rules, passed = lipinski_check(desc)

    lcol, rcol = st.columns([1, 2])
    with lcol:
        st.markdown(f"**Lipinski Rule of Five: {passed}/4 passed**")
        for rule, ok in rules.items():
            st.markdown(f"{'✅' if ok else '❌'} {rule}")
        if passed >= 3:
            st.success("Drug-like molecule")
        else:
            st.error("Poor drug-likeness")

    with rcol:
        st.dataframe({
            'Property': ['Molecular Weight', 'LogP', 'H-Bond Donors',
                         'H-Bond Acceptors', 'TPSA', 'Rotatable Bonds'],
            'Value':    [desc['MW'], desc['LogP'], desc['HBD'],
                         desc['HBA'], desc['TPSA'], desc['RotBonds']]
        }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.caption("Models: Random Forest (AUC 0.933) · Multi-task ChemBERTa (AUC 0.939) · Trained on MoleculeNet BBBP + ESOL")