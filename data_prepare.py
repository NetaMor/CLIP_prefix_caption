import numpy as np
import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import torch
import clip
import pandas as pd
import pickle
import json
import os
from tqdm import tqdm
import skimage.io as io

def transform_img(filename):
    #path_to_kor = '/home/dvir/Desktop/Projects/CLIP_prefix_caption/data/koronos/koronos_rgb/DJI_0026.JPG'
    try:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        transform1 = A.Compose([A.augmentations.crops.transforms.Crop(x_min=0, y_min=50, x_max=8000, y_max=6000, always_apply=False, p=1.0)])
        transform2 = A.Compose([A.augmentations.crops.transforms.CenterCrop (4280,5300, always_apply=False, p=1.0)])
        transformed = transform1(image=img)
        transformed = transform2(image=transformed['image'])
        #img = Image.fromarray(transformed['image'])
        #img.show()
    except:
        flag = 0
    return transformed


if __name__ == '__main__':
    data = pd.read_csv(f'./data/koronos/annotations.csv')
    data['image_id'] = data.image_id.apply(lambda x: 'DJI_'+'0'+str(int(x[-3:])-2) if len(str(int(x[-3:])-2))==3 else
    ('DJI_'+'00'+str(int(x[-3:])-2) if len(str(int(x[-3:])-2))==2 else ('DJI_'+'000'+str(int(x[-3:])-2) if len(str(int(x[-3:])-2))==1 else 'DJI_'+str(int(x[-3:])-2))))
    print("%0d captions loaded from json " % len(data))

    device = torch.device('cuda:0')
    clip_model_type = "ViT-B/32"
    clip_model_name = "ViT-B_32"
    out_path = f"./data/koronos/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    all_embeddings = []
    all_captions = []
    j = 0
    d = data[data.split == 'train'].reset_index(drop=True)
    for i in tqdm(range(len(d))):
            img_id = d.loc[i,'image_id']+".JPG"
            filename = f"./data/koronos/{img_id}"
            image = transform_img(filename)
            #image = io.imread(filename)
            #image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            image = preprocess(Image.fromarray(image['image'])).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            #d["clip_embedding"] = i
            all_embeddings.append(prefix)
            dic = {}
            dic['image_id'] = img_id
            dic['id'] = img_id
            dic['caption'] = d.loc[i,'caption']
            dic["clip_embedding"] = i

            all_captions.append(dic)
            j+=1
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))







