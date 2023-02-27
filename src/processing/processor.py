import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as tvF

import numpy as np
import nilearn.image
import skimage.measure
from PIL import Image
import cv2
from pytesseract import pytesseract

# pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
# !export TESSDATA_PREFIX= / usr / local / share /

import warnings

warnings.filterwarnings('ignore')

# from config import RESIZE
RESIZE = 768
class_mapping = {
    "bg": 0,
    "text": 1,
    "header": 2,
    0: 'bg',
    1: 'text',
    2: 'header'
}


def image_to_batch(img):
    img = img.resize(
        (RESIZE, RESIZE), resample=Image.Resampling.NEAREST)
    t = torch.tensor(np.array(img), dtype=float) / 255
    return torch.unsqueeze(torch.unsqueeze(t, 0), 0)


def split_masks_into_instances(t: torch.Tensor):
    masks = {}
    for class_num in range(masks.shape[0]):
        if class_mapping[class_num] == 'bg':
            continue
        t_class_map = t[class_num].numpy()
        components = skimage.measure.label(t_class_map)
        components_nums = np.unique(components)
        masks[class_num] = list()
        for cn in components_nums:
            if cn == 0:
                continue
            masks[class_num].append(
                (lambda x: x != cn)(components.copy())
            )

    return masks


def resample_masks(masks, target_shape):
    ret = nilearn.image.resample_img(
        img=masks,
        target_affine=(1, 2),
        target_shape=target_shape
    )
    return ret

class Processor:
    def __init__(self, model, device='cpu'):
        self.device = device
        model = model.to(device)
        self.model = model

    def get_model_predictions(self, x: torch.Tensor):
        def process_prediction(t):
            t = t[0]
            t = t.softmax(dim=0)
            t = t.argmax(dim=0)
            t = F.one_hot(t)
            t = t.transpose(0, 2).transpose(2, 1)
            t = t.bool()
            return t

        def process_input(x):
            return x[0]

        def process_input_and_prediction(x, t):
            x = process_input(x)
            t = process_prediction(t)
            x, t = x.to('cpu'), t.to('cpu')
            return x, t

        assert x.size() == (1, 1, RESIZE, RESIZE)
        x = x.to(self.device)
        with torch.no_grad():
            t = self.model(x)
        t = process_prediction(t)
        return t

    def process_img(self, img_path):
        img = Image.open(img_path)
        # x = np.asarray(img)
        t = self.get_model_predictions(image_to_batch(x))
        masks = split_masks_into_instances(resample_masks(t, x.shape))

