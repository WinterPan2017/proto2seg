from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd


class Cam16(Dataset):
    def __init__(self, df_list="", train=True, transform=None):
        all_df = pd.read_csv(df_list)
        train_df = all_df[0:80000]
        train_df.reset_index(drop=True, inplace=True)
        test_df = all_df[80000:100000]
        test_df.reset_index(drop=True, inplace=True)
        
        if train == True:
            self.list_df = train_df
        else:
            self.list_df = test_df

        self.length = self.list_df.shape[0]
        self.classes = [0, 1]
        self.targets = self.list_df['label']
            
        self.transform = transform
    def __getitem__(self, idx):
        
        impth = self.list_df['img'][idx]
        maskpth = self.list_df['mask'][idx]
        label = self.list_df['label'][idx]

        img = Image.open(impth)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length
