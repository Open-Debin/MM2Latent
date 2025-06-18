import sys
import os
sys.path.append("..")

import argparse
import torch

from lib import data_loader
import numpy as np
from torch.utils.data import DataLoader

device ="cuda" if torch.cuda.is_available() else "cpu"


def parser_parameters():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--ckph_allface", type=str, default="../models/greyseg_encoder/used_epoch495_train0p00124_val0p00114.ckpt", help="")

    parser.add_argument("--input_image_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")
    parser.add_argument("--save_file_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")
    
    parser.add_argument("--save_for_check", action="store_false")
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()
    args.old_new_name = ['.seg_gray_facerCeleb.png', '.seg_all_facerCeleb.npy']
    args.ckp_path = args.ckph_allface
    for arg_name, arg_value in vars(args).items():
        print(f"[{arg_name}]:\t{arg_value}")
    return args

def main():
    # Check if latents exist
    with torch.no_grad():
        sys.path.append(os.path.dirname(args.ckp_path))
        import model_arch
        model = model_arch.Autoencoder().load_from_checkpoint(args.ckp_path).to(device)
        model.eval()

    with torch.no_grad():
        with torch.inference_mode():
            img_list = data_loader.get_img_list(args.input_image_dir, indcator=args.old_new_name[0])
            dataset = data_loader.SegImageDataset(img_list, args.img_size, stage="inference")
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

            for images_path in dataloader:
                batch_data, data_path = images_path
                n_data = batch_data.shape[0]
                x = batch_data.to(device)
                
                # Saving input image and reconstruct image to check if is whole face or haironly
                # print losses to verify it.
                seg_codes = model.encoder(x).reshape((n_data, -1)).cpu().numpy()
                for seg_code, path2image in zip(seg_codes, data_path):
                    save_param_path = path2image.replace(args.input_image_dir, args.save_file_dir).replace(args.old_new_name[0], args.old_new_name[1])
                    np.save(save_param_path, seg_code)
                print(save_param_path)


if __name__ == "__main__":
    args = parser_parameters()
    main()




