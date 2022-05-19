import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/RSICD/oscar_split_{clip_model_name}_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/RSICD/annotations/dataset_rsicd.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    j = 0
    for i in tqdm(range(len(data["images"]))):
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
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))


#i didnt refer to all caption and didnt split val and train