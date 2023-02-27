import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import pathlib
import json
import fitz
import os
import numpy as np
from PIL import Image


def make_img_monocrome(img):
    img = img.min(dim=0).values
    foo = lambda x: 255 * (x>200)
    img.apply_(foo)
    return torch.stack([img, img, img], dim=0)

def create_train_dataset(datapath):
    return MyDatasetTrain(datapath)


def create_valid_dataset(datapath):
    return MyDatasetValid(datapath)

def create_extra_dataset():
    return MyDatasetValid()


def create_train_loader(train_dataset, batch_size=1):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return train_loader


def create_valid_loader(valid_dataset, batch_size=1):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return valid_loader


class MyDatasetTrain(Dataset):
    def __init__(self, DATA_PATH):
        self.path = DATA_PATH.joinpath('train')
        self.dirs = os.listdir(self.path)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        sub_dir_name = self.dirs[idx]
        img_file = self.path.joinpath(sub_dir_name, 'img.png')
        json_file = self.path.joinpath(sub_dir_name, 'description.json')
        with open(json_file, 'r') as f:
            description = json.load(f)

        img = torchvision.transforms.functional.pil_to_tensor(Image.open(img_file))
        img = make_img_monocrome(img)
        img = img / 255

        target = {}
        bboxes = torch.Tensor(description['bboxes'])
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor(description['labels'])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return img, target

class MyDatasetValid(Dataset):
    def __init__(self, DATA_PATH):
        self.path = DATA_PATH.joinpath('valid')
        self.dirs = os.listdir(self.path)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        sub_dir_name = self.dirs[idx]
        img_file = self.path.joinpath(sub_dir_name, 'img.png')
        json_file = self.path.joinpath(sub_dir_name, 'description.json')
        with open(json_file, 'r') as f:
            description = json.load(f)

        bboxes = description['bboxes']
        bboxes = torch.Tensor(bboxes)

        labels =  torch.as_tensor(description['labels'])
        img = torchvision.transforms.functional.pil_to_tensor(Image.open(img_file))
        img = torch.stack([img]*3, dim=1)[0]
        img = make_img_monocrome(img)
        img = img / 255

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return img, target
