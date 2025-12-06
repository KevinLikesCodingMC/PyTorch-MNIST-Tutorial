"""
MNIST Dataset Loader
Load CSV files
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=",", dtype=np.float32)
        self.labels = data[:, 0].astype(np.long)
        self.images = data[:, 1:] / 255.0
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # np arr -> torch tenser
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor
