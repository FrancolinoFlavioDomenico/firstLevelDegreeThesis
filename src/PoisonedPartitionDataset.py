import numpy as np
import torch
from torch.utils.data import Dataset

from globalVariable import seed_value
np.random.seed(seed_value)
torch.seed(seed_value)

class PoisonedPartitionDataset(Dataset):
    def __init__(self, dataset, possible_label):
        self.dataset = dataset
        self.possible_label = possible_label

    def __getitem__(self, index):
        scale = 0.8
        image, label = self.dataset[index]
        label = np.random.randint(0, self.possible_label) # label flipping
        noise = torch.randn(image.shape) * scale
        image = image + noise #img distorsion
        return image, label

    def __len__(self):
        return len(self.dataset)