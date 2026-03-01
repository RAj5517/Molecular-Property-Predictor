import streamlit as st
import torch
import pickle
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import sys, os

# import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.featurize import smiles_to_fingerprint, smiles_to_descriptors, lipinski_check
from src.utils import name_to_smiles
from src.model import MultiTaskChemBERTa

HF_REPO = "RAj5517/mol-property-predictor"

# ── Load all models (cached) ──
@st.cache_resource
def load_models():
    rf_bbbp_path = hf_hub_download(repo_id=HF_REPO, filename="rf_bbbp.pkl")
    rf_esol_path = hf_hub_download(repo_id=HF_REPO, filename="rf_esol.pkl")
    with open(rf_bbbp_path, 'rb') as f:
        rf_bbbp = pickle.load(f)
    with open(rf_esol_path, 'rb') as f:
        rf_esol = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    mt_path = hf_hub_download(repo_id=HF_REPO, filename="multitask_v3_best.pt")
    mt_model = MultiTaskChemBERTa()
    mt_model.load_state_dict(torch.load(mt_path, map_location='cpu'))
    mt_model.eval()

    return rf_bbbp, rf_esol, tokenizer, mt_model


def get_mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(300, 300))


def predict_rf(smiles, rf_bbbp, rf_esol):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None, None
    return rf_bbbp.predict_proba(fp.reshape(1,-1))[0][1], rf_esol.predict(fp.reshape(1,-1))[0]


def predict_multitask(smiles, tokenizer, mt_model):
    enc = tokenizer(smiles, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        bbbp_logit, esol_pred = mt_model(enc['input_ids'], enc['attention_mask'])
    return torch.sigmoid(bbbp_logit).item(), esol_pred.item()


# ── Page config ──
st.set_page_config(page_title="Molecular Property Predictor", page_icon="🧪", layout="wide")

# ── UI ──
st.title("🧪 Molecular Property Predictor")
st.markdown("Predict **Blood-Brain Barrier penetration** and **Solubility** for any molecule.")
st.markdown("---")

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

    smiles = user_input.strip()
    if Chem.MolFromSmiles(smiles) is None:
        with st.spinner("Looking up molecule..."):
            smiles = name_to_smiles(user_input.strip())

    if smiles is None:
        st.error("Could not find molecule. Try entering SMILES directly.")
        st.stop()

    st.success(f"SMILES: `{smiles}`")

    with st.spinner("Loading models..."):
        rf_bbbp, rf_esol, tokenizer, mt_model = load_models()

    desc  = smiles_to_descriptors(smiles)
    img   = get_mol_image(smiles)

    with st.spinner("Running predictions..."):
        rf_bbbp_prob, rf_esol_pred = predict_rf(smiles, rf_bbbp, rf_esol)
        mt_bbbp_prob, mt_esol_pred = predict_multitask(smiles, tokenizer, mt_model)

    st.markdown("---")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("🔬 Molecule")
        if img:
            st.image(img, width=280)
        st.caption(f"Input: {user_input}")

    with right:
        st.subheader("📊 Predictions")
        tab1, tab2 = st.tabs(["🧠 BBB Penetration", "💧 Solubility"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Random Forest", f"{rf_bbbp_prob:.1%}",
                          "Crosses BBB" if rf_bbbp_prob > 0.5 else "Does NOT cross")
            with c2:
                st.metric("Multi-task ChemBERTa", f"{mt_bbbp_prob:.1%}",
                          "Crosses BBB" if mt_bbbp_prob > 0.5 else "Does NOT cross")

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
                st.metric("Random Forest", f"{rf_esol_pred:.2f}", "log mol/L")
            with c2:
                st.metric("Multi-task ChemBERTa", f"{mt_esol_pred:.2f}", "log mol/L")

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