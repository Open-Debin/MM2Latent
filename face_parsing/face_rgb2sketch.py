import os
from PIL import Image
import numpy as np
import cv2

def create_sketch(image):

        t_lower = 100
        t_upper = 200 
        L2Gradient = True 
        image=np.uint8(image)
        edge = cv2.Canny(image, t_lower, t_upper, L2gradient = L2Gradient )

        return edge

root_path ="../data/CelebAHQ-500samples/"
save_path = "../data/CelebAHQ-500samples/"

for root, dirs, files in os.walk(root_path):
        for name in files:
            if '.jpg' in name:
                img_name = os.path.join(root, name)
                save_name = img_name.replace(root_path, save_path).replace('.jpg','.sketch.png')
            
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                image=Image.open(img_name).convert("RGB")
                image=np.array(image)
                image= cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
                edge = create_sketch(image)
                edge  =np.expand_dims(edge, axis=0)
                edge_image  =np.concatenate((edge,edge,edge),axis=0)
                edge_image=np.transpose(edge_image,[1,2,0])
                edge_image=np.uint8(edge_image)
                sketch= Image.fromarray(edge_image)
                sketch.save(save_name)
                
