import math
import os
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import uniform_filter

C1 = 0.0001 * 4
C2 = 0.0009 * 4


def make_dataset(dir):
    Himgs, Limgs = [],[]
    for organ in os.listdir(dir):
        Hpath = os.path.join(dir,organ,'high_quality')
        Lpath = os.path.join(dir,organ,'low_quality')
        Hfnames = sorted(os.listdir(Hpath))
        Lfnames = sorted(os.listdir(Lpath))
        for fname in Hfnames:
            path = os.path.join(Hpath, fname)
            Himgs.append(path)
        for fname in Lfnames:
            path = os.path.join(Lpath, fname)
            Limgs.append(path)
    return Limgs, Himgs


class USDataset(Dataset):
    def __init__(self, Limgs, Himgs, data_len=-1, image_size=[256, 256]):
        if data_len > 0:
            self.Limgs, self.Himgs = Limgs[:int(data_len)], Himgs[:int(data_len)]
        else:
            self.Limgs, self.Himgs = Limgs, Himgs
        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5, inplace=True)
        ])
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        Limg, Himg = Image.open(self.Limgs[index]), Image.open(self.Himgs[index])
        ret['Limg'], ret['Himg'] = self.tfs(Limg), self.tfs(Himg)
        return ret

    def __len__(self):
        return len(self.Limgs)


def make_Cdataset(dir):
    Himgs, Limgs = [],[]
    for i, organ in enumerate(sorted(os.listdir(dir))): # ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
        Hpath = os.path.join(dir,organ,'high_quality')
        Lpath = os.path.join(dir,organ,'low_quality')
        Hfnames = sorted(os.listdir(Hpath))
        Lfnames = sorted(os.listdir(Lpath))
        for fname in Hfnames:
            path = os.path.join(Hpath, fname)
            Himgs.append({'path': path, 'class': i})
        for fname in Lfnames:
            path = os.path.join(Lpath, fname)
            Limgs.append({'path': path, 'class': i})
    return Limgs, Himgs


class CDataset(Dataset):
    def __init__(self, Limgs, Himgs, image_size=[256, 256]):
        self.Limgs, self.Himgs = Limgs, Himgs
        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5, inplace=True)
        ])
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        Limg, Himg = Image.open(self.Limgs[index]['path']), Image.open(self.Himgs[index]['path'])
        ret['Limg'], ret['Himg'], ret['class'] = self.tfs(Limg), self.tfs(Himg), self.Limgs[index]['class']
        return ret

    def __len__(self):
        return len(self.Limgs)


def ssim(x, y):
    x_mean = x.mean(dim=(2, 3))
    y_mean = y.mean(dim=(2, 3))
    x_var = x.var(dim=(2, 3), correction=0)
    y_var = y.var(dim=(2, 3), correction=0)
    covar = (x * y).mean(dim=(2, 3)) - x_mean * y_mean
    ssim_map = ((2 * x_mean * y_mean + C1) * (2 * covar + C2)) / (
                (x_mean ** 2 + y_mean ** 2 + C1) * (x_var + y_var + C2))
    return ssim_map.mean()


def lncc(input1, input2):
    mean1 = torch.mean(input1)
    mean2 = torch.mean(input2)
    std1 = torch.std(input1)
    std2 = torch.std(input2)
    cross_correlation = torch.mean((input1 - mean1) * (input2 - mean2))
    ncc = cross_correlation / (std1 * std2)
    return ncc






    

