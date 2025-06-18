
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np

idx2category = {0: 'background', 0.1: 'face', 0.2: 'rb', 0.3: 'lb', 0.4: 're', 0.5: 'le', 0.6: 'nose', 0.7: 'ulip',
                0.8: 'imouth', 0.9: 'llip', 1: 'hair'}
category2idx = {'background': 0, 'face': 1, 'rb': 2, 'lb': 3, 're': 4, 'le': 5, 'nose': 6, 'ulip': 7, 'imouth': 8,
                'llip': 9, 'hair': 1}




class SegImageDataset(Dataset):
    def __init__(self, img_list, img_size=128, stage='train'):
        self.img_list = img_list
        if stage == "train":
            self.transform = transforms.Compose([

                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=15, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),

            ]
            )
        elif stage == "inference":
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),

            ])
        else:
            raise ValueError("stage should be either train, inference")
        pass
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return (img, img_path)


class SketchImageDataset(Dataset):
    def __init__(self, img_list, img_size=128, stage='train'):
        self.img_list = img_list
        if stage == "train":
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=15, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize((0), (250 / 255))

            ]
            )
        elif stage == "inference":
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize((0), (250 / 255))

            ])
        else:
            raise ValueError("stage should be either train, inference")
        pass
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return (img, img_path)



class load_demo_data(Dataset):
    def __init__(self, img_dir, img_name_list, stage='train'):
        img_list = []
        for img_name in img_name_list:
            full_path2img = os.path.join(img_dir, img_name)
            img_list.append(full_path2img)
        self.img_list = img_list

        if stage == "train":
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            )
        elif stage == "inference":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0), (250 / 255))
            ])
        else:
            raise ValueError("stage should be either train, inference")
        # pdb.set_trace()
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seg_img_path = self.img_list[idx]
        seg_img = Image.open(seg_img_path)
        seg_img = self.transform(seg_img)

        rgb_img = self.transform(Image.open(seg_img_path.replace('.seg_gray_facerCeleb.png', '.jpg')))
        sketch_img = self.transform(Image.open(seg_img_path.replace('.seg_gray_facerCeleb.png', '.sketch.png')))

        seg_embed = torch.tensor(np.load(seg_img_path.replace('.seg_gray_facerCeleb.png', '.seg_all_facerCeleb.npy')))
        sketch_emb = torch.tensor(np.load(seg_img_path.replace('.seg_gray_facerCeleb.png', '.sketch.npy')))

        return (rgb_img, seg_img, sketch_img, seg_embed, sketch_emb)


def get_img_list(root2latents, indcator='seg.png'):
    img_list = []
    for root, dirs, files in os.walk(root2latents):
        for name in files:
            if name.endswith(indcator):
                full_path2img = os.path.join(root, name)
                img_list.append(full_path2img)
    return img_list


class TextLoader(Dataset):
    def __init__(self, prompts_file, clip_farl, clip_vitb32, device):
        with open(prompts_file, 'r') as rfile:
            self.prompt_ls = [e.strip() for e in rfile.readlines()]

        with torch.no_grad():
            self.text_tokens = clip.tokenize(self.prompt_ls).to(device)
            text_features_farl = clip_farl.encode_text(self.text_tokens).float()
            self.text_features_farl = (text_features_farl / text_features_farl.norm(dim=-1, keepdim=True)).cpu()

            text_features_vitb32 = clip_vitb32.encode_text(self.text_tokens).float().cpu()
            self.text_features_vitb32 = (text_features_vitb32 / text_features_vitb32.norm(dim=-1, keepdim=True)).cpu()

            self.text_tokens = self.text_tokens.cpu()

    def __len__(self):
        return len(self.prompt_ls)

    def __getitem__(self, index):
        return self.text_features_farl[index], self.text_features_vitb32[index]

    def show_data(self):
        for idx, txt in enumerate(self.prompt_ls):
            print(f'{idx + 1}\t{txt}')



