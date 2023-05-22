from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

class BCSS(Dataset):
    def __init__(self, df_list = '', train=True, transform=None):
        all_df = pd.read_csv(df_list)
        train_df = all_df[all_df['split']=='train']
        train_df.reset_index(drop=True, inplace=True)
        train_df = train_df[:20000]
        test_df = all_df[all_df['split']=='test']
        test_df.reset_index(drop=True, inplace=True)
        test_df = train_df[:20000]
        self.transform = transform
        
        if train == True:
            self.list_df = train_df
        else:
            self.list_df = test_df

        self.length = self.list_df.shape[0]
            

    def __getitem__(self, idx):
        
        impth = self.list_df['patch_path'][idx]
        mskpth = self.list_df['mask_path'][idx]

        img = Image.open(impth)
        msk = Image.open(mskpth)

        msk_fin = np.array(msk)
        msk_ori = np.array(msk)
        msk_fin[msk_ori==0] = 0
        msk_fin[msk_ori==1] = 1
        msk_fin[msk_ori==2] = 2
        msk_fin[msk_ori==3] = 3
        msk_fin[msk_ori==4] = 4
        msk_fin[msk_ori==5] = 0
        msk_fin[msk_ori==6] = 0
        msk_fin[msk_ori==7] = 0
        msk_fin[msk_ori==8] = 0
        msk_fin[msk_ori==9] = 0
        msk_fin[msk_ori==10] = 3
        msk_fin[msk_ori==11] = 3
        msk_fin[msk_ori==12] = 0
        msk_fin[msk_ori==13] = 0
        msk_fin[msk_ori==14] = 0
        msk_fin[msk_ori==15] = 0
        msk_fin[msk_ori==16] = 0
        msk_fin[msk_ori==17] = 0
        msk_fin[msk_ori==18] = 0
        msk_fin[msk_ori==19] = 1
        msk_fin[msk_ori==20] = 1
        msk_fin[msk_ori==21] = 0
        
        target = np.array([(msk_fin == i).sum() for i in range(5)])
        target = target / target.sum()
        label = np.argmax(target) if target.max() > 0.8 else 5
        # label = np.argmax(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, msk_fin, label

    def __len__(self):
        return self.length