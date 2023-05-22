import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import pandas as pd


class WSIDataset(Dataset):
    def __init__(self, df_list, train=True, transform=None):
        self.list_df = pd.read_csv(df_list)
        self.key = ""
        if "BCSS" in df_list:
            self.key = "patch_path"
        else:
            self.key = "img"
        self.length = self.list_df.shape[0]
            

    def __getitem__(self, idx):
        
        impth = self.list_df[self.key][idx]

        img = Image.open(impth)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.length