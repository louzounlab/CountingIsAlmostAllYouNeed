from torch.utils.data import DataLoader, Dataset
import torch
"""
Dataset for each repertoire

inputs:
- data: The reactive TCR numbers within each repertoire
- labels: the labels of the repertoires
"""

class RepertoireDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx])
        label = self.labels[idx]
        return sample, label
