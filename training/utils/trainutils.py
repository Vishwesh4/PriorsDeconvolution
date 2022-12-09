"""
This file contains all the important functions needed for loading model and dataset
"""
from typing import Union, Tuple, Any

import numpy as np
import torch
import torchmetrics
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import gaussian
from tqdm import tqdm
import wandb

from .dataloader import BlurLoader
import trainer

@trainer.Metric.register("ce_prior")
class Test_Metric(trainer.Metric):
    def get_metrics(self):
        metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(), torchmetrics.ConfusionMatrix(num_classes=2)]
        )
        return metricfun


@trainer.Dataset.register("ce_prior")
class Blur_Dataset(trainer.Dataset):
    def get_transforms(self) -> Tuple[Any, Any]:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale()
            ]
        )
        return transform, transform

    def get_loaders(self):
        trainset = BlurLoader(blur_kernel=self.fspecial_gaussian_2d((self.kwargs["blur_size"], self.kwargs["blur_size"]), self.kwargs["blur_sigma"]),
                sigma=self.kwargs["sigma"],
                image_pth=self.path,
                tile_h=self.kwargs["tile_h"],
                tile_w=self.kwargs["tile_w"],
                tile_stride_factor_h=self.kwargs["tile_stride_factor_h"],
                tile_stride_factor_w=self.kwargs["tile_stride_factor_w"],
                mode="train",
                mask_pth=self.kwargs["mask_pth"],
                lwst_level_idx=self.kwargs["lwst_level_idx"],
                threshold=self.kwargs["threshold"],
                transform=self.train_transform)
        
        testset = BlurLoader(blur_kernel=self.fspecial_gaussian_2d((self.kwargs["blur_size"], self.kwargs["blur_size"]), self.kwargs["blur_sigma"]),
                sigma=self.kwargs["sigma"],
                image_pth=self.path,
                tile_h=self.kwargs["tile_h"],
                tile_w=self.kwargs["tile_w"],
                tile_stride_factor_h=self.kwargs["tile_stride_factor_h"],
                tile_stride_factor_w=self.kwargs["tile_stride_factor_w"],
                mode="test",
                mask_pth=self.kwargs["mask_pth"],
                lwst_level_idx=self.kwargs["lwst_level_idx"],
                threshold=self.kwargs["threshold"],
                transform=self.test_transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.train_batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.test_batch_size, shuffle=True
        )
        return trainset, trainloader, testset, testloader

    @staticmethod
    def fspecial_gaussian_2d(size, sigma):
        kernel = np.zeros(tuple(size))
        kernel[size[0]//2, size[1]//2] = 1
        kernel = gaussian(kernel, sigma)
        return kernel/np.sum(kernel)


@trainer.Logger.register("ce_prior")
class CE_logger(trainer.Logger):
    def log_table(self, input, output, label, epoch):
        columns = ["id", "image", "real class", "calculated class"]
        table = wandb.Table(columns=columns)
        _, preds = torch.max(output.data, 1)
        for i in range(2):
            idx = f"{epoch}_{i}"
            image = wandb.Image(input[i].permute(1, 2, 0).cpu().numpy())
            table.add_data(idx, image, preds[i], label[i])
        self.log({"table_key": table})


@trainer.Model.register("ce_prior")
class CE_model(trainer.Model):
    def __init__(self,**kwargs):
        super().__init__()
        self.model = torchvision.models.__dict__["resnet18"](pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        output = self.model(x)
        return output


class TrainEngine(trainer.Trainer):
    def train(self):
        self.model.train()
        for data in tqdm(self.dataset.trainloader):
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.loss_fun(outputs, label)
            loss.backward()
            self.optimizer.step()
            # Track loss
            self.logger.track(loss_value=loss.item())
            # metric calculation
            self.metrics(outputs, label)
            # Logging loss
            self.logger.log({"Epoch Train loss": loss.item()})
        self.metrics.compute()
        self.metrics.log()
        print(
            "Total Train loss: {}".format(
                np.mean(self.logger.get_tracked("loss_value"))
            )
        )

    def val(self):
        self.model.eval()
        for data in tqdm(self.dataset.testloader):
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            outputs = self.model(image)
            loss = self.loss_fun(outputs, label)
            # Track loss
            self.logger.track(loss_value=loss.item())
            # metric calculation
            self.metrics(outputs, label)
            # Logging loss
            self.logger.log({"Epoch Train loss": loss.item()})
        self.metrics.compute()
        self.metrics.log()
        if self.current_epoch % 5 == 0:
            self.logger.log_table(image, outputs, label, self.current_epoch)

        mean_loss = np.mean(self.logger.get_tracked("loss_value")) / len(
            self.dataset.testloader
        )
        print("Total Val loss: {}".format(mean_loss))

        return self.metrics.results["val_Accuracy"], mean_loss