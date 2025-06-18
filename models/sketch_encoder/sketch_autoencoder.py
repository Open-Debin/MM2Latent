import sys

sys.path.append("../..")
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.utils import save_image
device = "cuda" if torch.cuda.is_available() else "cpu"


class SketchAutoencoderFrameWork(pl.LightningModule):
    def __init__(self, lr=1e-2, val_savepath='', decoder_type='naive', onehot=True, act_type='lrelu'):
        super().__init__()
        # Encoder
        self.lr = lr
        self.encoder = NaiveEncoder(act_type)
        self.decoder = NaiveDecoder(act_type)

        self.onehot = onehot
        self.criterion = nn.CrossEntropyLoss() if self.onehot else nn.MSELoss()
        self.val_savepath = val_savepath

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        if len(batch) == 1:
            x = batch[0]
        else:
            x, _ = batch
            label = x[:, 0, :, :]

        x = label[:,None]
        y = self(x)
        loss = self.criterion(y, label.long())
        self.log_dict({'train_loss': loss, 't_mse': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        label = x[:, 0, :, :]
        x = label[:,None]
        y = self(x)
        loss = self.criterion(y, label.long())
        self.log_dict({'val_loss': loss, 'v_mse': loss})
        # self.log('val_loss', loss)
        if batch_idx % 1 == 0:
            # Get a sample of 4 input images and their corresponding reconstructions
            epoch_num = self.current_epoch
            x_sample = x[:4]
            y_sample = y[:4].argmax(dim=1).unsqueeze(1) if self.onehot else y[:4]
            # Concatenate the samples along the batch dimension

            sample = torch.cat([x_sample, y_sample], dim=0)

            x_embed = self.encoder(x)
            n_batch = x_embed.shape[0]
            x_embed = x_embed.reshape((n_batch, -1))

            save_path = f'{self.val_savepath}/epoch-{epoch_num}/validation_imgsize{x.shape[-1]}_{batch_idx}_embedNorm{torch.norm(x_embed, dim=1).mean().item()}.png'
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            save_image(sample, save_path, nrow=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


#=====================#
#==  Conv block ==#
#=====================#

''' 
Conv 
'''
class NaiveEncoder(nn.Module):
    def __init__(self, act_type='lrelu'):
        super(NaiveEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)  
        elif act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError('act_type should be lrelu or silu, but your import is {act_type}') 

        self.naivenet1 = NaiveBlock(16, 16, mode='encoder', act_type=act_type)
        self.naivenet2 = NaiveBlock(16, 32, mode='encoder', act_type=act_type)
        self.naivenet3 = NaiveBlock(32, 32, mode='encoder', act_type=act_type)
        self.naivenet4 = NaiveBlock(32, 16, mode='encoder', act_type=act_type)
        self.naivenet5 = NaiveBlock(16, 8, mode='encoder', act_type=act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.naivenet1(x)
        x = self.naivenet2(x)
        x = self.naivenet3(x)
        x = self.naivenet4(x)
        x = self.naivenet5(x)

        return x

class NaiveDecoder(nn.Module):
    def __init__(self, act_type='lrelu'):
        super(NaiveDecoder, self).__init__()

        self.naivenet1 = NaiveBlock(8, 16, mode='decoder', act_type=act_type)
        self.naivenet2 = NaiveBlock(16, 32, mode='decoder', act_type=act_type)
        self.naivenet3 = NaiveBlock(32, 32, mode='decoder', act_type=act_type)
        self.naivenet4 = NaiveBlock(32, 16, mode='decoder', act_type=act_type)
        self.naivenet5 = NaiveBlock(16, 16, mode='decoder', act_type=act_type)

        self.final_conv1 = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x = self.naivenet1(x)
        x = self.naivenet2(x)
        x = self.naivenet3(x)
        x = self.naivenet4(x)
        x = self.naivenet5(x)

        x = self.final_conv1(x)
        return x

class NaiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mode='encoder', act_type='lrelu'):
        super(NaiveBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)  
        elif act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError('act_type should be lrelu or silu, but your import is {act_type}') 

        if mode == 'encoder':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        elif mode == 'decoder':
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        else:
            raise ValueError('Report error')

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out)
        return out

