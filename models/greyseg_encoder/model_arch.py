import sys

sys.path.append("../..")
import os
import pdb

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchvision.utils import save_image



device = "cuda" if torch.cuda.is_available() else "cpu"
from pytorch_msssim import  SSIM, MS_SSIM

idx2category = {0: 'background', 0.1: 'face', 0.2: 'rb', 0.3: 'lb', 0.4: 're', 0.5: 'le', 0.6: 'nose', 0.7: 'ulip',
                0.8: 'imouth', 0.9: 'llip', 1: 'hair'}
category2idx = {'background': 0, 'face': 1, 'rb': 2, 'lb': 3, 're': 4, 'le': 5, 'nose': 6, 'ulip': 7, 'imouth': 8,
                'llip': 9, 'hair': 1}


class Autoencoder(pl.LightningModule):
    def __init__(self, lr=1e-3, val_savepath='', flag_hair_only=False):
        super().__init__()
        # Encoder
        self.lr = lr
        channel = 1
        self.encoder = nn.Sequential(
            self.encoder_block(channel, 16),
            self.encoder_block(16, 32),
            self.encoder_block(32, 32),
            self.encoder_block(32, 16),
            self.encoder_block(16, 8),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)

        self.val_savepath = val_savepath
        self.num_ls = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.flag_hair_only = flag_hair_only

    def encoder_block(self, in_c, out_c):

        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True)
        )

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def weighted_l1_loss(self, input, target, weight):
        return torch.mean(weight * torch.abs(input - target))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # x = self.random_remove_seg_layer(x)
        if self.flag_hair_only:
            x = self.hair_only(x)

        y = self(x)
        loss = self.mse_loss(y, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        # x = self.random_remove_seg_layer(x)
        if self.flag_hair_only:
            x = self.hair_only(x)

        y = self(x)
        loss = self.mse_loss(y, x)
        self.log('val_loss', loss)
        if batch_idx % 1 == 0:
            # Get a sample of 4 input images and their corresponding reconstructions
            epoch_num = self.current_epoch
            x_sample = x[:4]
            y_sample = y[:4]
            # Concatenate the samples along the batch dimension
            sample = torch.cat([x_sample, y_sample], dim=0)
            # Rescale the pixel values from [-1, 1] to [0, 1] for saving as an image
            # Save the sample image to disk
            save_path = f'{self.val_savepath}epoch-{epoch_num}/validation_imgsize{x.shape[-1]}_{batch_idx}.png'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            save_image(sample, save_path, nrow=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def random_remove_seg_layer(self, x):
        select_layer = self.num_ls[torch.randint(0, len(self.num_ls), (torch.randint(0, 10, (1,)),))]
        for item in select_layer:
            mask_select = torch.where(x == item, torch.tensor(1.).to(device), torch.tensor(0.0).to(device))
            x[mask_select == 1] = 0
        return x

    def hair_only(self, x):
        mask_select = torch.where(x == category2idx['hair'], torch.tensor(1.).to(device), torch.tensor(0.0).to(device))
        x[mask_select == 0] = 0
        return x

