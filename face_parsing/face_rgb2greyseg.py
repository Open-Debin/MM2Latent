
import os

import argparse
import time
from PIL import Image
import torch
import numpy as np
import copy
import facer

device = "cuda" if torch.cuda.is_available() else "cpu"

def parser_parameters():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--old_new_name", type=list, default=[".jpg", '.seg_gray_facerCeleb.png'], help="")
    parser.add_argument("--input_image_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")
    parser.add_argument("--save_file_dir", type=str, default= "../data/CelebAHQ-500samples/", help="")

    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        print(f"[{arg_name}]:\t{arg_value}")
    return args

def get_maxface_id(faces_rects, h_img=1024, w_img=1024):
    area = []
    for content in faces_rects:
        x1, y1, x2, y2 = content
        y1, y2 = [max(min(v, h_img-1), 0) for v in [y1, y2]]
        x1, x2 = [max(min(v, w_img-1), 0) for v in [x1, x2]]
        area.append(torch.tensor((x2 - x1) * (y2 -y1)))
    max_area = torch.argmax(torch.tensor(area))
    return max_area.item()

def select_max_detface(faces_det, max_id):
    faces_det = copy.deepcopy(faces_det)
    faces_det['image_ids'] = faces_det['image_ids'][max_id][None]
    faces_det['rects'] = faces_det['rects'][max_id][None]
    faces_det['points'] = faces_det['points'][max_id][None]
    faces_det['scores'] = faces_det['scores'][max_id][None]
    return faces_det

def show_sec2hour_minutes_seconds(duration_seconds):
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = int(duration_seconds % 60)
    # print out the duration in hours, minutes, and seconds
    print("Function duration: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))


def convert_grayimg_to_one_hot(input_tensor, num_classes=19):
    # Create a tensor to hold the one-hot encoding
    one_hot_tensor = torch.zeros((input_tensor.size(0), num_classes, *input_tensor.size()[2:]), dtype=torch.float32)

    # Iterate through each unique value and create a binary mask
    for i in range(num_classes):
        value_mask = (input_tensor == i * 10).float()
        one_hot_tensor[:, i, :, :] = value_mask.view_as(one_hot_tensor[:, i, :, :])

    return one_hot_tensor

def convert_probmatrix_to_one_hot(probability_tensor):
    _, max_indices = torch.max(probability_tensor, dim=1, keepdim=True)
    one_hot_tensor = torch.zeros_like(probability_tensor)
    one_hot_tensor.scatter_(1, max_indices, 1)
    return one_hot_tensor


def main():

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_parser = facer.face_parser('farl/celebm/448', device=device)

    with torch.inference_mode():
        for root, dirs, files in os.walk(args.input_image_dir):
            print(root)

            start_time = time.time()
            files = files[::-1]
            for name in files:
                if args.old_new_name[0] in name:
                    full_path2img = os.path.join(root, name)
                    save_seg_path = full_path2img.replace(args.input_image_dir, args.save_file_dir).replace(args.old_new_name[0], args.old_new_name[1])
                    if os.path.exists(save_seg_path):
                        print(f'skip:{save_seg_path}')
                        continue
                    image_dim4 = facer.hwc2bchw(facer.read_hwc(full_path2img)).to(device=device)
                    try:
                        faces_det = face_detector(image_dim4)
                    except:
                        continue
                    try:
                        faces_det['image_ids']
                    except:
                        continue
                    if len(faces_det['image_ids']) > 1:
                        print("max_face_detect")
                        max_id = get_maxface_id(faces_det['rects'].clone().detach())
                        faces_det = select_max_detface(faces_det, max_id)

                    faces_parse = face_parser(image_dim4, faces_det)
                    seg_logits = faces_parse['seg']['logits']
                    seg_map = seg_logits.argmax(dim=1)*10
                    seg_np = seg_map[0].cpu().numpy().astype('uint8')

                    # Here .png is 12k, while .jpg is 42k. but image quality .png > .jpgß
                    save_dir = os.path.dirname(save_seg_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    set_PIL = Image.fromarray(seg_np)
                    set_PIL.save(save_seg_path) 

                    duration_seconds = time.time() - start_time
                    # Testing codes
                    sum_val = (np.asarray(set_PIL) != np.asarray(Image.open(save_seg_path))).sum()
                    if sum_val != 0:
                        print("error")
                        break

                    print(np.unique(np.asarray(set_PIL)))
                    print(np.unique(Image.open(save_seg_path)))
                    print(f"error {sum_val}, duration_seconds: {duration_seconds}")

            
            # convert duration to hours, minutes, and seconds
            show_sec2hour_minutes_seconds(duration_seconds)


if __name__ == "__main__":
    args = parser_parameters()
    main()