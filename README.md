## Trained Models

Weights hosted on HuggingFace — no training required.

🤗 https://huggingface.co/RAj5517/mol-property-predictor

| File | Description | Metric |
|------|-------------|--------|
| chemberta_best.pt | ChemBERTa fine-tuned on BBBP | AUC: 0.9339 |
| rf_bbbp.pkl | Random Forest — BBB penetration | AUC: 0.9330 |
| rf_esol.pkl | Random Forest — Solubility | R²: 0.71 |


## Results

| Model | BBBP AUC | ESOL RMSE |
|-------|----------|-----------|
| Random Forest | 0.9330 | 1.163 |
| ChemBERTa single-task | 0.9339 | — |
| Multi-task ChemBERTa V3 | **0.9393** | **0.821** |

Multi-task model beats both baselines simultaneously.
One model predicts BBB penetration AND solubility.