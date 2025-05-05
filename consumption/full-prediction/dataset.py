import torch
from torch.utils.data import Dataset


class ResStockSetpointPredictionDataset(Dataset):
    def __init__(self, data, labels):
        self.X = data
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
