---
title: Molecular Property Predictor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.54.0
app_file: app/app.py
pinned: false
---

# 🧪 Molecular Property Predictor

A machine learning system that predicts **Blood-Brain Barrier (BBB) penetration** and **aqueous solubility** for drug-like molecules. Built with RDKit, ChemBERTa, and multi-task learning — trained on real experimental data from MoleculeNet.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/RAj5517/mol-property-predictor)  
**Model Weights:** [HuggingFace Hub](https://huggingface.co/RAj5517/mol-property-predictor)

---

## 🎯 What This Project Does

Given a molecule name (e.g. "Aspirin") or a SMILES string, the app predicts:

- **Will it cross the Blood-Brain Barrier?** — critical for neurological drug development
- **How soluble is it in water?** — critical for drug absorption and bioavailability
- **Is it drug-like?** — Lipinski Rule of Five check with all descriptors

---

## 🧠 Why These Properties Matter

### Blood-Brain Barrier (BBB) Penetration
The brain is protected by a highly selective barrier that blocks most molecules. A drug targeting the brain — for depression, Alzheimer's, epilepsy, pain — **must** cross this barrier. If it cannot cross, it never reaches the brain and is useless for that indication.

```
Examples:
  Caffeine    → crosses BBB ✓  (why coffee affects your brain)
  Morphine    → crosses BBB ✓  (brain pain receptor agonist)
  Penicillin  → does NOT cross (cannot treat brain infections directly)
  Aspirin     → does NOT cross (pain relief via peripheral nerves)
```

### Aqueous Solubility (ESOL)
Your body is mostly water. A drug must dissolve to be absorbed into the bloodstream. Poor solubility = drug passes through the gut without being absorbed = therapeutic failure.

```
Solubility scale (log mol/L):
   0 to -1    → highly soluble   (like sugar)
  -1 to -3    → moderately soluble (like Aspirin: -3.05)
  -3 to -5    → poorly soluble   (concerning)
  below -5    → very poorly soluble (likely development failure)
```

---

## 📊 Results

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| Random Forest (baseline) | BBBP | ROC-AUC | 0.9330 |
| ChemBERTa (fine-tuned) | BBBP | ROC-AUC | 0.9339 |
| **Multi-task ChemBERTa** | **BBBP** | **ROC-AUC** | **0.9393** |
| Random Forest (baseline) | ESOL | RMSE | 1.163 |
| **Multi-task ChemBERTa** | **ESOL** | **RMSE** | **0.821** |

**Key finding:** Multi-task learning beats both single-task models on both properties simultaneously. One shared model outperforms separate specialized models — demonstrating the power of shared molecular representations.

---

## 🏗️ Architecture

### Three-Model Progression

The project follows a deliberate progression of increasing sophistication:

#### Model 1 — Random Forest (Baseline)
```
SMILES → RDKit → Morgan Fingerprint (2048-bit) → Random Forest → BBB label / solubility
```
- Morgan fingerprints capture local chemical substructures (radius=2, 2048 bits)
- 100 decision trees, each trained on a random molecule subset
- No GPU needed — trains in seconds
- Establishes a strong baseline: **AUC 0.933**

#### Model 2 — ChemBERTa Fine-tuning
```
SMILES → ChemBERTa tokenizer → [CLS] embedding (768-dim) → Classification head → BBB label
```
- ChemBERTa pre-trained on **77 million** molecules from PubChem via masked language modeling
- Fine-tuned on 1631 BBBP training molecules (transfer learning)
- Optimal: 5 epochs, lr=2e-5, AdamW
- Improves baseline: **AUC 0.9339**

#### Model 3 — Multi-task ChemBERTa (Final)
```
SMILES → ChemBERTa backbone (shared) → BBB head → BBB label
                                     → ESOL head → solubility value
```
- Single shared backbone learns general molecular features
- Two task-specific heads trained simultaneously
- Alternating gradient updates: one BBBP batch, one ESOL batch per step
- Best performance: **AUC 0.9393** (BBB) · **RMSE 0.821** (solubility)

### Why Multi-task Works Better
```
BBB penetration and solubility are not independent.
Both depend on:
  - Lipophilicity (LogP)
  - Molecular size
  - H-bond donors/acceptors
  - Polar surface area

Sharing a backbone means:
  Signal from 1128 ESOL molecules helps learn BBB patterns
  Signal from 2039 BBBP molecules helps learn solubility patterns
  Total effective training signal: 3167 molecules instead of 2039
```

---

## 📁 Project Structure

```
molecular-property-predictor/
│
├── app/
│   └── app.py              ← Streamlit web application
│
├── notebooks/
│   ├── 01_baseline_rf.ipynb        ← Day 1: Random Forest training
│   ├── chemberta_finetune.ipynb    ← Day 2: ChemBERTa fine-tuning
│   └── 03_multitask_model.ipynb    ← Day 3: Multi-task training
│
├── src/
│   ├── featurize.py        ← Morgan fingerprints, RDKit descriptors, Lipinski
│   ├── model.py            ← MultiTaskChemBERTa architecture
│   ├── dataset.py          ← PyTorch Dataset classes (BBBP + multi-task)
│   ├── evaluate.py         ← ROC-AUC, RMSE, R² evaluation functions
│   └── utils.py            ← PubChem API lookup, common molecule dictionary
│
├── results/                ← Training metrics and comparison charts
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧬 Datasets

### BBBP — Blood Brain Barrier Penetration
- **Source:** MoleculeNet (experimental measurements from published literature)
- **Size:** 2039 molecules
- **Task:** Binary classification (1 = crosses BBB, 0 = does not)
- **Class balance:** 76% positive, 24% negative (imbalanced → use ROC-AUC, not accuracy)
- **Split:** 80/20 train/test, stratified

### ESOL — Estimated Solubility
- **Source:** MoleculeNet / Delaney dataset (measured in lab, published 2004)
- **Size:** 1128 molecules
- **Task:** Regression (log solubility in mol/L)
- **Range:** -0.07 to -11.6 log mol/L
- **Split:** 80/20 train/test, random

All measurements are real experimental values — not computed or estimated.

---

## 🔬 Technical Details

### Molecular Representation — Morgan Fingerprints

```python
# Each molecule → 2048-bit binary vector
# Bit i = 1 means "this substructure exists in the molecule"

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = generator.GetFingerprintAsNumPy(mol)   # shape: (2048,)
```

**How Morgan fingerprints work:**
- For each atom, look at all atoms within radius=2 bonds
- Hash that local chemical environment to a position in 2048-bit vector
- Two similar molecules → similar fingerprints → similar Random Forest predictions

### ChemBERTa — Chemistry Language Model

ChemBERTa is a RoBERTa model pre-trained on 77 million SMILES strings from PubChem using masked language modeling — the same pre-training as BERT but on chemistry instead of English text.

```
Pre-training (done by authors):
  Input:  "CC(=O)Oc1ccccc[MASK]c1C(=O)O"
  Target: predict masked atom token
  Data:   77,000,000 molecules from PubChem

Fine-tuning (done by us):
  Input:  "CC(=O)Oc1ccccc1C(=O)O"  (Aspirin)
  Target: 0 (does not cross BBB)
  Data:   1631 BBBP training molecules
```

The [CLS] token embedding (768 dimensions) represents the entire molecule — this is what we feed into the classification/regression heads.

### Multi-task Training Loop

```python
for epoch in range(EPOCHS):
    for bbbp_batch, esol_batch in zip(bbbp_loader, esol_loader):
        
        # BBBP step — classification
        bbbp_logits, _ = model(bbbp_input_ids, bbbp_attention_mask)
        loss_bbbp = BCEWithLogitsLoss(bbbp_logits, bbbp_labels)
        loss_bbbp.backward()
        optimizer.step()
        
        # ESOL step — regression  
        _, esol_preds = model(esol_input_ids, esol_attention_mask)
        loss_esol = MSELoss(esol_preds, esol_labels)
        loss_esol.backward()
        optimizer.step()
```

Alternating gradient updates ensure the backbone learns features useful for both tasks.

### Hyperparameters (Final Multi-task Model)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning rate | 1.5e-5 | Between too-fast (2e-5 overfits) and too-slow (1e-5 underfits) |
| Epochs | 15 | Model still improving at epoch 9; plateaus around epoch 12 |
| Optimizer | AdamW | Weight decay regularization |
| Weight decay | 0.01 | Prevent overfitting on small dataset |
| Grad clip | 1.0 | Stabilize training |
| Batch size | 32 | Memory efficient on T4 GPU |
| Head size | 128 | Sufficient capacity without overfitting |

---

## ⚠️ Limitations

### Prediction Accuracy

The model achieves **ROC-AUC 0.9393** on the test set — meaning roughly 1 in 15 predictions 
is incorrect. We tested only a handful of molecules manually; real-world error rate on 
unseen molecules may vary.

Known cases where the model is likely to be wrong:

**Molecules actively pumped out by efflux transporters (P-glycoprotein)**
These look drug-like on paper but are actively removed from the brain by membrane pumps.
Morgan fingerprints have no way to encode this dynamic biological process.
Examples: Penicillin, Imatinib, Digoxin, Vinblastine

**Large, charged, or highly polar molecules**
The model was trained mostly on small organic molecules.
Peptides, biologics, and highly charged molecules are underrepresented in BBBP.

**Molecules very different from the training set**
2039 training molecules is a small sample of drug-like chemical space.
Novel scaffolds or unusual chemistries may not generalize.

**What the fingerprints miss:**
- Active efflux and influx transporters
- Molecular charge at physiological pH (pKa effects)  
- 3D conformation and shape
- Plasma protein binding
- Metabolic stability

### What This Means

AUC 0.939 does not mean 93.9% of individual predictions are correct.
It means: given one BBB-positive and one BBB-negative molecule,
the model ranks them correctly 93.9% of the time.
For any single molecule, treat the output as a probability estimate — not a verdict.

**These predictions are for early-stage computational screening only.
Always validate with in vitro BBB assays and in vivo pharmacokinetic studies.**
```

### Dataset size

2039 molecules is small by deep learning standards. The model has not seen enough chemical diversity to generalize perfectly. Real pharma models train on millions of proprietary compounds.

### Label quality

BBBP dataset labels come from different experimental protocols across different labs. Some labels may be inconsistent or condition-dependent (dose, species, route of administration).

**These predictions are probabilistic guides for early-stage screening — not definitive clinical answers. Always validate with wet lab experiments.**

---

## 🚀 Getting Started

### Local Setup

```bash
# clone repo
git clone https://github.com/RAj5517/Molecular-Property-Predictor.git
cd Molecular-Property-Predictor

# create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# install dependencies
pip install -r requirements.txt

# run app
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

### Usage

**By molecule name:**
```
Type: Caffeine
→ App calls PubChem API → gets SMILES automatically
→ Predicts BBB + solubility
```

**By SMILES string:**
```
Type: Cn1cnc2c1c(=O)n(c(=O)n2C)C
→ App parses SMILES directly
→ Predicts BBB + solubility
```

### Loading Trained Models

```python
from huggingface_hub import hf_hub_download
from src.model import MultiTaskChemBERTa
from transformers import AutoTokenizer
import torch

REPO = "RAj5517/mol-property-predictor"

# load multi-task model
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = MultiTaskChemBERTa()
weights_path = hf_hub_download(repo_id=REPO, filename="multitask_v3_best.pt")
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

# predict
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
enc = tokenizer(smiles, return_tensors='pt', padding='max_length',
                truncation=True, max_length=128)
with torch.no_grad():
    bbbp_logit, esol_pred = model(enc['input_ids'], enc['attention_mask'])

bbbp_prob = torch.sigmoid(bbbp_logit).item()
solubility = esol_pred.item()
print(f"BBB probability: {bbbp_prob:.1%}")
print(f"Log solubility:  {solubility:.2f} mol/L")
```

---

## 🤗 Model Weights

All weights hosted on HuggingFace — no training required to run the app.

**Repository:** https://huggingface.co/RAj5517/mol-property-predictor

| File | Description | Size | Performance |
|------|-------------|------|-------------|
| `multitask_v3_best.pt` | Multi-task ChemBERTa (final) | 177MB | AUC 0.9393 · RMSE 0.821 |
| `chemberta_best.pt` | Single-task ChemBERTa | 176MB | AUC 0.9339 |
| `rf_bbbp.pkl` | Random Forest BBB classifier | 4MB | AUC 0.9330 |
| `rf_esol.pkl` | Random Forest solubility regressor | 8MB | RMSE 1.163 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Cheminformatics | RDKit |
| Molecular fingerprints | Morgan fingerprints (2048-bit, radius=2) |
| Pre-trained model | ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) |
| Deep learning | PyTorch + HuggingFace Transformers |
| Classical ML | scikit-learn RandomForest |
| Data | MoleculeNet (BBBP + ESOL) |
| App framework | Streamlit |
| Model hosting | HuggingFace Hub |
| Compute | Google Colab (T4 GPU) |

---

## 📚 References

- **MoleculeNet:** Wu et al. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*
- **ChemBERTa:** Chithrananda et al. (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. *NeurIPS Workshop*
- **BBBP Dataset:** Martins et al. (2012). A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling. *Journal of Chemical Information and Modeling*
- **ESOL Dataset:** Delaney (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *Journal of Chemical Information and Computer Sciences*
- **Morgan Fingerprints:** Morgan (1965). The Generation of a Unique Machine Description for Chemical Structures. *Journal of Chemical Documentation*
- **RDKit:** Landrum (2006). RDKit: Open-source cheminformatics. https://www.rdkit.org

---

## 📈 Training Environment

- **Compute:** Google Colab (Free tier, NVIDIA Tesla T4 GPU)
- **Training time:** ~20 min per model
- **Python:** 3.12
- **Key packages:** rdkit 2025.9.5, transformers 5.x, torch 2.x, scikit-learn 1.6

---

## 🔮 Future Work

- Add **Tox21** dataset (12 toxicity endpoints) for multi-task expansion
- Add **ClinTox** (clinical trial toxicity) prediction
- Implement **similarity search** — find FDA-approved drugs similar to query molecule
- Add **3D conformation** prediction using ETKDG algorithm
- Train on **larger proprietary datasets** to improve generalization
- Add **attention visualization** — which atoms does ChemBERTa focus on?
- **Drug repurposing** — given a target protein, screen approved drugs for binding

---

## 👤 Author

**Sayanendu** — [@RAj5517](https://github.com/RAj5517)

Built as part of an AI/ML portfolio focused on medical and chemistry applications.

---

*Predictions are for research and educational purposes only. Not for clinical use.*