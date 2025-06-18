import torch
import torch.nn as nn
import pytorch_lightning as pl
from lib.models_arch import EqualizedLinear


class BiPathModel(pl.LightningModule):
    
    def __init__(self, path1_dim=512, path2_dim=512):
        super().__init__()
        self.map_fn = BiPathMappingNet(text_dim=path1_dim, path2_dim=path2_dim)

    def forward(self, x):
        return self.map_fn(x)


class BiPathMappingNet(nn.Module):
    """
    <a id="mapping_network"></a>
    ## Mapping Network
    ![Mapping Network](mapping_network.svg)
    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, text_dim:int, path2_dim:int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        features = 512
        n_layers = 11
        input_dim = text_dim + path2_dim
        self.text_dim = text_dim
        
        layers = []
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Dropout())
        layers.append(EqualizedLinear(input_dim, features))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor):
        w_delta = self.net(z)
        return w_delta
