import numpy as np
import torch
from torch.utils.data import Dataset


class VQVAELatents(Dataset):
    def __init__(
        self,
        path,
    ):
        super().__init__()

        self.latents = np.load(path)[0].transpose(1, 2, 0)
        self.h, self.w, self.c = self.latents.shape

        self.coords_horizontal = (np.arange(self.w) + 0.5) / self.w * 2 - 1
        self.coords_vertical = (np.arange(self.h) + 0.5) / self.h * 2 - 1

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, idx):

        i = idx // self.w
        j = idx % self.w

        return {"coords": torch.from_numpy([self.coords_horizontal[i], self.coords_vertical[j]]).float()}, {
            "latents": self.latents[i, j],
        }

