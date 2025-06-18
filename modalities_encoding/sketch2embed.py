import os
import sys
sys.path.append("..")
import pdb
import torch
import argparse

from lib import data_loader
import numpy as np
from torch.utils.data import DataLoader

device ="cuda" if torch.cuda.is_available() else "cpu"



def parser_parameters():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--ckp_path", type=str, default="../models/sketch_encoder/used_epoch458_train0p11285_val0p1277.ckpt")

    parser.add_argument("--input_image_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")
    parser.add_argument("--save_file_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")
    parser.add_argument("--save_for_check", action="store_false")
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()
    args.old_new_name = ['.sketch.png', '.sketch.npy']

    for arg_name, arg_value in vars(args).items():
        print(f"[{arg_name}]:\t{arg_value}")
    return args


def main():
    with torch.no_grad():
        sys.path.append(os.path.dirname(args.ckp_path))
        import sketch_autoencoder
        model = sketch_autoencoder.SketchAutoencoderFrameWork().load_from_checkpoint(args.ckp_path).to(device)
        model.eval()

    with torch.no_grad():
        img_list = data_loader.get_img_list(args.input_image_dir, indcator=args.old_new_name[0])
        dataset = data_loader.SketchImageDataset(img_list, args.img_size, stage="inference")
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=16)

        for images_path in dataloader:
            batch_data, data_path = images_path
            n_data = batch_data.shape[0]
            x = batch_data.to(device)
            x = x[:, 0, :, :][:,None]
            

            seg_codes = model.encoder(x).reshape((n_data, -1)).cpu().numpy()
            for seg_code, path2image in zip(seg_codes, data_path):
                save_param_path = path2image.replace(args.input_image_dir, args.save_file_dir).replace(args.old_new_name[0], args.old_new_name[1])
                save_dir = os.path.dirname(save_param_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if len(seg_code) != 512:
                    raise ValueError(f'length of seg_code error')

                # # Check if the saved file is correct
                tmp = np.load(save_param_path)
                mse = np.mean((tmp - seg_code)**2)
                print(f"mse: {mse}")
                pdb.set_trace()
                np.save(save_param_path, seg_code)
            print(save_param_path)


if __name__ == "__main__":
    args = parser_parameters()
    main()




