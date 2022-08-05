from torch.utils.data import Dataset
import torch
import numpy as np
import librosa
from icecream import ic


class CausalDataset(Dataset):
    def __init__(self, t, x_continuous, x_categorical, y):
        super().__init__()

        assert t.shape[0] == x_continuous.shape[0]
        assert x_categorical.shape[0] == y.shape[0]
        assert t.shape[0] == y.shape[0]

        self.x_continuous = x_continuous.copy()
        self.x_categorical = x_categorical.copy()
        self.y = y.copy()
        self.t = t.copy()

    def __len__(self):
        return self.x_continuous.shape[0]

    @staticmethod
    def convert_to_tensor(x_continuous, x_categorical, y, t):
        x_continuous = torch.from_numpy(x_continuous).requires_grad_(True).type(torch.FloatTensor)
        x_categorical = torch.from_numpy(x_categorical).requires_grad_(True).type(torch.FloatTensor)
        t = torch.from_numpy(np.array(t)).requires_grad_(True).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.FloatTensor)

        return x_continuous, x_categorical, y, t

    def __getitem__(self, idx):
        x_categorical = self.x_categorical[idx]
        x_continuous = self.x_continuous[idx]
        t = self.t[idx]
        y = np.array(self.y[idx], dtype=np.float)

        x_continuous, x_categorical, y, t = self.convert_to_tensor(x_continuous, x_categorical, y, t)
        return x_continuous, x_categorical, y, t
