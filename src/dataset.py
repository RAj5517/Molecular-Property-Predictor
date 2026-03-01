import torch
from torch.utils.data import Dataset

class BBBPDataset(Dataset):
    """Dataset for BBBP classification task."""
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles    = smiles_list
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning (BBBP + ESOL)."""
    def __init__(self, smiles_list, bbbp_label=None, esol_label=None, max_length=128, tokenizer=None):
        self.smiles    = smiles_list
        self.bbbp      = bbbp_label
        self.esol      = esol_label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'bbbp_label':     torch.tensor(self.bbbp[idx], dtype=torch.float) if self.bbbp else torch.tensor(-1.0),
            'esol_label':     torch.tensor(self.esol[idx], dtype=torch.float) if self.esol else torch.tensor(float('nan')),
        }