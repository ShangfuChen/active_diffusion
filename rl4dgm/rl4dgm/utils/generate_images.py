
import argparse
import os 
import csv
import numpy as np
import random

import torch 
from diffusers import DiffusionPipeline

def generate_images(
    model, 
    img_save_dir, 
    prompt, 
    n_images,
    generator,
    n_inference_steps=10,
    img_dim=(256,256),
):
    # create image save folder
    os.makedirs(img_save_dir, exist_ok=False)

    n_imgs_saved = 0
    while n_imgs_saved < n_images:
        print("n images saved", n_imgs_saved)
        n_imgs_per_prompt = n_images - n_imgs_saved
        images = model( # list of PIL.Images
            prompt,
            num_inference_steps=n_inference_steps,
            generator=generator,
            num_images_per_prompt=n_imgs_per_prompt,
            height=img_dim[0],
            width=img_dim[1],
        ).images
        
        for im in images:
            # Get raw pixel values to see if the image is black (black image is generated if NSFW content is detected). Only save if it is not a black image
            if np.all(np.array(im.getdata()) == [0,0,0]):
                print("NSFW black image generated. Not saving this image.")
            else:
                im.save(os.path.join(img_save_dir, f"{n_imgs_saved}.jpg"), "JPEG")
                n_imgs_saved += 1
            
    print(f"Generated {n_imgs_saved} images")

def generate_cat_images(
    model,
    img_save_dir,
    n_images,
    generator,
    n_inference_steps=10,
    img_dim=(256,256),
):

    prompts = [
        "Cute black cat with round eyes",
        "Demonic black cat with sharp eyes",
        "Cute white cat with round eyes",
        "Demonic white cat with sharp eyes",
    ]

    # Compute number of images to generate per prompt
    n_prompts = len(prompts)
    base_int = n_images // n_prompts
    remainder = n_images % n_prompts
    n_imgs_per_prompt = [base_int] * n_prompts
    for i in range(remainder):
        n_imgs_per_prompt[i] += 1
    random.shuffle(n_imgs_per_prompt)

    for prompt, n in zip(prompts, n_imgs_per_prompt):
        img_save_folder = os.path.join(img_save_dir, prompt.replace(" ", "_"))
        generate_images(
            model=model,
            img_save_dir=img_save_folder,
            prompt=prompt,
            n_images=n,
            generator=generator,
            n_inference_steps=n_inference_steps,
            img_dim=img_dim,
        )
    