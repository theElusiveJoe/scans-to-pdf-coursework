import os
import pathlib
import json

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np


class PagesDataset(Dataset):
    def __init__(self, resize, datadir, classes_list, class_num_mapping, orig_class_name, maxlen):
        self.datadir = datadir
        self.len = len(os.listdir(datadir)[:maxlen+1 if maxlen > 0 else None])

        self.classes_list = classes_list
        self.class_mapping = class_num_mapping
        self.classes_num = len(self.classes_list)
        self.orig_class_name = orig_class_name

        self.resize = resize

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        def img_to_tensor(img):
             return torch.tensor(np.array(img)).float()

        folder = pathlib.Path(self.datadir).joinpath(str(index))

        img = Image.open(folder.joinpath(self.orig_class_name+'.bmp')).resize(
            (self.resize, self.resize),
            resample=Image.Resampling.NEAREST
        )

        masks = [
            Image.open(
                folder.joinpath(self.class_mapping[classnum]+'.bmp')
            ).resize(
                (self.resize, self.resize),
                resample=Image.Resampling.NEAREST
            )

            for classnum in sorted(self.class_mapping.keys())
        ]

        masks = list(map(img_to_tensor, masks))
        masks = torch.stack(masks, dim=0)
        return (
            torch.unsqueeze(
                img_to_tensor(img),
                0
            )/255,
            masks/255
        )


def create_dataloaders(resize, batch_size, datadir, classes_list, class_num_mapping, orig_class_name, max_images):
    dataset = PagesDataset(
        resize=resize,
        datadir=datadir,
        classes_list=classes_list,
        class_num_mapping=class_num_mapping,
        orig_class_name=orig_class_name,
        maxlen=max_images
    )

    test_num = int(0.1 * len(dataset))
    train_num = len(dataset) - test_num
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_num, test_num],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return {
        'train': train_dataloader,
        'valid': test_dataloader
    }