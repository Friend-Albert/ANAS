import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ValidData(Dataset):
    def __init__(self, path, device=torch.device('cuda')):
        self.path = path
        self.device = device
        df = pd.read_csv(path)
        data = df.to_numpy()
        if len(data) % 42 != 0:
            print("Warning: 数据长度不能被42整除，需要调整")
        n_samples = len(data) // 42
        self.data = data[:n_samples * 42].reshape(n_samples, 42, -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item][:, :12]
        target = self.data[item][0, 12]
        data = torch.tensor(data).float().to(self.device)
        target = torch.tensor(target).unsqueeze(dim=0).float().to(self.device)
        return data, target


if __name__ == "__main__":
    dataset = ValidData("../tmp/validation_set.csv")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    for i, (data, target) in enumerate(dataloader):
        print(data.shape, target.shape)
