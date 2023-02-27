import torch

import matplotlib.pyplot as plt

import time
from tqdm.auto import tqdm

from src.faster_rcnn.utils.model_saver import ModelSaver

from src.faster_rcnn.datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.model_saver = None
        self.model = None

    def train_model(self, model, name):
        train_dataset = create_train_dataset(self.config.DATA_PATH)
        valid_dataset = create_valid_dataset(self.config.DATA_PATH)
        self.train_loader = create_train_loader(train_dataset)
        self.valid_loader = create_valid_loader(valid_dataset)
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}\n")

        self.model = model.to(self.DEVICE)

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005
        )

        self.train_loss_iter_list = []
        self.val_loss_iter_list = []
        self.train_loss_avg_list = []
        self.val_loss_avg_list = []

        self.model_saver = ModelSaver(
            path_to_save=self.config.TRAINED_MODELS_PATH,
            name=name
        )

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEPOCH {epoch + 1} of {self.config.NUM_EPOCHS}")
            start = time.time()

            train_loss_avg = self.train_one_epoch()
            val_loss_avg = self.validate()
            self.train_loss_avg_list.append(train_loss_avg)
            self.val_loss_avg_list.append(val_loss_avg)

            print(f"Epoch #{epoch + 1} train loss: {train_loss_avg:.3f}")
            print(f"Epoch #{epoch + 1} validation loss: {val_loss_avg:.3f}")

            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}")

            self.model_saver(self.model, val_loss_avg)

    def train_one_epoch(self):
        # print('Training')
        losses_sum = 0
        prog_bar = tqdm(self.train_loader, total=len(self.train_loader))
        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()
            images, targets = data

            images = [image.to(self.DEVICE) for image in images]
            keys = list(targets.keys())
            nums = len(targets[keys[0]][0])
            targets = [
                {key: targets[key][0].to(self.DEVICE) for key in list(targets.keys())} for num in range(len(targets[keys[0]][0]))
            ]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            self.train_loss_iter_list.append(loss_value)
            losses_sum += loss_value

            losses.backward()
            self.optimizer.step()

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return losses_sum/len(self.train_loader)

    def validate(self):
        # print('Validating')
        losses_sum = 0
        prog_bar = tqdm(self.valid_loader, total=len(self.valid_loader))
        for i, data in enumerate(prog_bar):
            images, targets = data

            images = [image.to(self.DEVICE) for image in images]
            keys = list(targets.keys())
            nums = len(targets[keys[0]][0])
            targets = [
                {key: targets[key][0].to(self.DEVICE) for key in list(targets.keys())} for num in range(len(targets[keys[0]][0]))
            ]

            with torch.no_grad():
                loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            self.val_loss_iter_list.append(loss_value)
            losses_sum += loss_value

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return losses_sum/len(self.valid_loader)

    def draw_per_epoch_plot(self):
        x = range(len(self.train_loss_avg_list))
        y1 = self.train_loss_avg_list
        y2 = self.val_loss_avg_list
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.show()