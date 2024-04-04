"""
Preference functions to provide AI feedback
"""
import os

import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image

"""
Dummy preference function that prefers images with higher red pixel values

Arg:
    prompt (str) : prompt corresponding to the image pair
    images (list(PILImage)) : a pair of images with the same prompt
"""
def ColorPickOne(**kwargs):
    im0, im1 = kwargs['images']
    im0, im1 = np.array(im0), np.array(im1)
    if im0[:, :, 0].sum() > im1[:, :, 0].sum():
        label0 = 1
        label1 = 0
    else:
        label0 = 0
        label1 = 1
    return (label0, label1)


def ColorScoreOne(**kwargs):
    im = np.array(kwargs['images'][0])
    score = im[:, :, 0].mean() - im[:, :, 1:].mean()
    return 1+(score / 255 * 9)


def PickScore(**kwargs):
    model = kwargs['model']
    im = np.array(kwargs['images'][0])
    score = im[:, :, 0].mean() - im[:, :, 1:].mean()
    return 1+(score / 255 * 9)


def calc_probs(processor, model, images, prompt, device):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        try:
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        except:
            image_embs = model.module.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.module.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.module.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist(), scores.cpu().tolist()


"""
Use finetuned pickscore to calculate rewards
"""
def pickscore():
    def _fn(processer, model, images, prompts, device):
        if isinstance(images, torch.Tensor):
            imgs = []
            toPIL = torchvision.transforms.ToPILImage()
            for im in images:
                imgs.append(toPIL(im))    
        _, score = calc_probs(processer, model, imgs, prompts, device)
        return score, {}
    return _fn
