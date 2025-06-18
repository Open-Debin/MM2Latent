import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


import numpy as np
import importlib
device ="cuda" if torch.cuda.is_available() else "cpu"


def show_image(image, figsize=(320,320)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def tensor2image_dim4(image_tensor):
    '''
    Inverse output of GAN tensor[-1,1] to [0, 1]
    output shape: [num, 3, size,size]
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().clamp_(0, 1)
    return image_unflat


def tensor2grid_image(image_unflat, num_images=16, nrow=3):
    '''
    input shape: [num, 3, size,size]
    put images [num, 3, size,size] into image grid.
    
    '''
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    return image_grid.permute(1, 2, 0).squeeze()
    

@torch.no_grad()
def clipfeature2image_tensor(img_features, map_fn, generator, flag_small_memory=True):
    '''
    flag_mean_f is the preprocess, flag_norm_f is the old version now we unsupport it.
    so the pipeline is img_features => mean_f_preprocess => F2W_mapFullyNetwork => denorm_w(wplus) => GAN_Generator => image(output) .
    '''

    wplus = map_fn(img_features)
    if len(wplus) == 2:
        wplus = wplus[0]

    if flag_small_memory:
        image_tensor = []
        for wplus_vec in wplus:
            image, _ = generator([wplus_vec.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=False)
            image_tensor.append(image)
        return torch.cat(image_tensor)
    else:
        image_norm, _ = generator([wplus], input_is_latent=True, randomize_noise=False, return_latents=False)
        return image_norm




def load_module(module_root, module_name):
    module_path = os.path.join(module_root, f"{module_name}.py")
    module_name = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(module_name)
    module_name.loader.exec_module(module)
    return module



