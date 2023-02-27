import config
import os
os.chdir(config.PROJECT_ROOT_PATH)

import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

from src.unet import dataset, loss, unet, unet

import GPUtil

import numpy as np
import skimage.measure
from PIL import Image
import cv2
from pytesseract import pytesseract


import warnings
warnings.filterwarnings('ignore')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, dataloaders, n_epochs):
    train_dataloader, valid_dataloader = dataloaders['train'], dataloaders['valid']

    results = {
        'train': [],
        'valid': []
    }

    for epoch in range(1, n_epochs + 1):
        GPUtil.showUtilization()
        print(f'EPOCH {epoch} STARTED')
        # training
        model.train()
        train_losses = []
        for batch_i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred_mask = model(x)
            loss = criterion(pred_mask, y)
            print('epoch:', epoch, 'train batch', batch_i, 'loss:', loss.item())
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # testing
        model.eval()
        valid_losses = []
        for batch_i, (x, y) in enumerate(valid_dataloader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred_mask = model(x)
                loss = criterion(pred_mask, y)
                print('epoch:', epoch, 'valid batch', batch_i, 'loss:', loss.item())
            valid_losses.append(loss.item())

        # draw_results(x,pred_mask)

        results['train'].append(np.mean(train_losses))
        results['valid'].append(np.mean(valid_losses))
        print(results['train'][-1], results['valid'][-1])

        # torch.save(model.state_dict(), f'trained_models/epoch_{epoch}_state.st')
    return results, model


RESIZE = 512
# -1, если нужно использовать все возможные картинки
MAX_IMAGES_NUM = 100
N_EPOCHS = 1
BATCH_SIZE = 1

dataset_params = {
    'resize':RESIZE,
    'batch_size':BATCH_SIZE,
    'datadir':config.DATA_PATH,
    'classes_list':config.CLASSES_LIST,
    'class_num_mapping':config.NUMS_CLASSES,
    'orig_class_name':config.ORIG_CLASS_NAME,
    'max_images':MAX_IMAGES_NUM
}

dataloaders = dataset.create_dataloaders(
    **dataset_params
)

model = unet2.UNet2(len(config.CLASSES_LIST))
model.to(device)

criterion = loss.JaccardLoss('multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

results, model = train_model(model, criterion, optimizer, dataloaders, n_epochs=N_EPOCHS)