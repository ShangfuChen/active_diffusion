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
    return 1+(im[:, :, 0].mean() / 255 * 9)
