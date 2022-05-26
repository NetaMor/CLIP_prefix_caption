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



if __name__ == '__main__':
    device = torch.device('cuda:0')
    #clip_model_type = "ViT-B/32"
    #clip_model_name = "ViT-B_32"
    #out_path = f"./data/koronos/oscar_split_{clip_model_name}_train.pkl"
    #clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    data = pd.read_csv(f'./data/koronos/annotations.csv')
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    j = 0
    for i in tqdm(range(len(data))):
        d = data["images"][i]
        if d['split']=='val':
            img_id = d["filename"]
            filename = f"./data/RSICD/RSICD_images/{img_id}"
            if not os.path.isfile(filename):
                filename = f"./data/RSICD/RSICD_images/{img_id}"
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            #d["clip_embedding"] = i
            all_embeddings.append(prefix)
            dic = {}
            dic['image_id'] = img_id
            dic['id'] = d['imgid']
            dic['caption'] = d['sentences'][0]['raw']
            dic["clip_embedding"] = j

            all_captions.append(dic)
            j+=1
            if (i + 1) % 10000 == 0:
                with open(out_path, 'wb') as f:
                    pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))



    path_to_kor = '/home/dvir/Desktop/Projects/CLIP_prefix_caption/data/koronos/koronos_rgb/DJI_0026.JPG'
    img = cv2.imread(path_to_kor)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)

    path_to_black = '/home/dvir/Desktop/Projects/CLIP_prefix_caption/data/koronos/DJI_0027.JPG'
    img_black = cv2.imread(path_to_black)
    img_black = cv2.cvtColor(img_black, cv2.COLOR_BGR2RGB)

    print(img_black.shape)

    transform = A.Compose([A.augmentations.crops.transforms.CenterCrop (int(img_black.shape[0]),int(img_black.shape[1]), always_apply=False, p=1.0)])
    transformed = transform(image=img)
    img = Image.fromarray(transformed['image'])
    img.show()




