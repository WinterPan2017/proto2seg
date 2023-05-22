import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from pathlib import Path
import shutil
import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.mat = torch.zeros((num_classes, num_classes),
                               dtype=torch.int64, device=device)

    def update(self, a, b):
        n = self.num_classes
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def miou(self):
        h = self.mat.float()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iou.mean().item() * 100

    def ious(self):
        h = self.mat.float()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iou

    def dice(self):
        h = self.mat.float()
        iou = torch.diag(h)*2 / (h.sum(1) + h.sum(0))
        return iou

    def acc(self):
        h = self.mat.float()
        acc = torch.diag(h) / h.sum(1)
        return acc

    def global_acc(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        return acc

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )
