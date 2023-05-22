import torch
import cv2
import os
import csv
import numpy as np
from torch.utils import data
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf


class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, mask, angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        return image, mask

    def flip(self, image, mask):
        if random.random() > 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random() < 0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        return image, mask

    def randomCrop(self, image, mask, size=512):
        resize_size = size
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=[size, size])
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_contrast(image, factor)
        #mask = tf.adjust_contrast(mask,factor)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_brightness(image, factor)
        #mask = tf.adjust_contrast(mask, factor)
        return image, mask

    def centerCrop(self, image, mask, size=512):
        if size == None:
            size = image.size
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        return image, mask

    def adjustSaturation(self, image, mask):
        factor = transforms.RandomRotation.get_params([0.8, 1.2])
        image = tf.adjust_saturation(image, factor)
        #mask = tf.adjust_saturation(mask, factor)
        return image, mask

    # scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
    def randomResizeCrop(self, image, mask, scale=(0.3, 1.0), ratio=(1, 1)):
        msk = np.array(mask)
        h_image, w_image = msk.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size,
                               interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask

    def randomGrayscale(self, image, mask, p=0.2):
        if random.random() < p:
            image = tf.rgb_to_grayscale(image, num_output_channels=3)
        return image, mask


class CamelyonDataset(data.Dataset):
    """Camelyon Dataset"""
    num_classes = 2

    def __init__(self,
                 split_file='',
                 dataset_path='',
                 train=True,
                 prototype_mask_folder=None,
                 return_gt=False):
        self.dataset_path = dataset_path
        self.split_file = split_file
        self.prototype_mask_folder = prototype_mask_folder
        self.return_gt = return_gt
        self.train = train

        all_df = pd.read_csv(split_file)
        train_df = all_df[all_df['type'] == 'train']
        train_df.reset_index(drop=True, inplace=True)
        test_df = all_df[all_df['type'] == 'valid']
        test_df.reset_index(drop=True, inplace=True)

        if self.train:
            self.list_df = train_df
        else:
            self.list_df = test_df

        self.length = self.list_df.shape[0]

    def __getitem__(self, idx):
        image_path = self.dataset_path+'train_patchs/' + \
            self.list_df['slide_id'][idx]+'/images/' + \
            self.list_df['image_id'][idx]+'.png'
        if not self.return_gt:
            annotation_path = os.path.join(
                self.prototype_mask_folder, self.list_df['slide_id'][idx] + '_' + self.list_df['image_id'][idx]+'.png')
        else:
            annotation_path = self.dataset_path+'train_patchs/' + \
                self.list_df['slide_id'][idx]+'/masks/' + \
                self.list_df['image_id'][idx]+'.png'
        img = Image.open(image_path)
        msk = Image.open(annotation_path)

        if self.train is True:
            aug = Augmentation()
            img, msk = aug.flip(img, msk)
            img, msk = aug.adjustContrast(img, msk)
            img, msk = aug.adjustBrightness(img, msk)
            img, msk = aug.adjustSaturation(img, msk)
            img, msk = aug.randomGrayscale(img, msk)
            img, msk = aug.randomResizeCrop(img, msk)

        img = tf.to_tensor(img)
        msk = np.array(msk)
        msk[msk > 0] = 1
        msk = torch.from_numpy(msk).to(dtype=torch.long)
        return img, msk

    def __len__(self):
        return self.length

