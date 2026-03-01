## Trained Models

Weights hosted on HuggingFace — no training required.

🤗 https://huggingface.co/RAj5517/mol-property-predictor

| File | Description | Metric |
|------|-------------|--------|
| chemberta_best.pt | ChemBERTa fine-tuned on BBBP | AUC: 0.9339 |
| rf_bbbp.pkl | Random Forest — BBB penetration | AUC: 0.9330 |
| rf_esol.pkl | Random Forest — Solubility | R²: 0.71 |

## Results

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| Random Forest | BBBP | ROC-AUC | 0.9330 |
| ChemBERTa | BBBP | ROC-AUC | 0.9339 |
| Random Forest | ESOL | R² | 0.71 |