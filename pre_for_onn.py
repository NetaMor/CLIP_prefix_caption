import clip
import os
from torch import nn
import pickle
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image
from torchvision import transforms

#import cog

# import torch

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device("cpu")



class Predictor():
    def __init__(self,path_weights_model):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda")
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.prefix_length = 10
        model = ClipCaptionModel(self.prefix_length)
        model.load_state_dict(torch.load(path_weights_model, map_location=CPU))#'./pretrained_models/'+weights_path, map_location=CPU))
        model = model.eval()
        model = model.to(self.device)
        self.model = model

    def forward(self, images):
        """Run a single prediction on the model"""
        images_pre = []
        model = self.model
        for image in images:
            pil_image = PIL.Image.fromarray(torch.squeeze(image).numpy())
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            images_pre.append(image)
        images_pre = torch.cat(images_pre, dim=0)
        prefix_embed_list = []
        with torch.no_grad():
            prefix = self.clip_model.encode_image(images_pre).to(
                self.device, dtype=torch.float32
            )
            for i in range(prefix.shape[0]):
                prefix_embed = model.clip_project(prefix[i,:]).reshape(1, self.prefix_length, -1)
                prefix_embed_list.append(prefix_embed)
            prefix_embed_tensor = torch.cat(prefix_embed_list,dim=0)
        """if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)"""
        return prefix_embed_tensor


class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def preprocess_clip(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/RSICD/oscar_split_{clip_model_name}_test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/RSICD/annotations/dataset_rsicd.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    all_images = []
    j = 0
    for i in tqdm(range(len(data["images"]))):
        d = data["images"][i]
        if d['split']=='test':
            img_id = d["filename"]
            filename = f"./data/RSICD/RSICD_images/{img_id}"
            if not os.path.isfile(filename):
                filename = f"./data/RSICD/RSICD_images/{img_id}"
            image = io.imread(filename)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            d["clip_embedding"] = i
            all_embeddings.append(prefix)
            #all_embeddings.append(image)
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
    path_to_images = []
    for file in os.listdir("./data/koronos/koronos_rgb"):
        UPLOADED_FILE = os.path.join("./data/koronos/koronos_rgb", file)
        path_to_images.append(UPLOADED_FILE)
    images = []
    for image in path_to_images:
        image = io.imread(image)
        image = image[np.newaxis,:]
        images.append(torch.tensor(image))
    images_ten = torch.cat(images,dim=0)


    model_pre = Predictor('./checkpoints/rsicd_prefix_GPT_30epoch/rsicd_prefix_GPT_30epoch-010.pt')
    model_pre.forward(images_ten)

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    preprocess_clip(args.clip_model_type)