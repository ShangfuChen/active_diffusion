
import argparse
import os 
import csv
import numpy as np
import random
import shutil

import torch 
from diffusers import DiffusionPipeline

class ImageGenerator:

    def __init__(self,):
        self.TASK_TO_PROMPTS = {
            "cat" : [
                "Cute black cat with round eyes",
                "Demonic black cat with sharp eyes",
                "Cute white cat with round eyes",
                "Demonic white cat with sharp eyes",
            ],
            "ice cream" : [
                "Premium soft serve ice cream",
                "The world's most expensive soft serve ice cream",
                "The world's most expensive soft serve ice cream, topped with edible gold and jewerly-like fruits",
                "The world's most expensive soft serve ice cream, topped with edible gold and a tiara decorated by jewerly-like fruits",
            ],
        }

    def generate_images(
        self,
        model, 
        img_save_dir, 
        prompt, 
        n_images,
        generator,
        n_inference_steps=10,
        img_dim=(256,256),
    ):
        # create image save folder
        if os.path.exists(img_save_dir):
            # remove directory if it already exists
            shutil.rmtree(img_save_dir)
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

    def generate_images_from_prompt_list(
        self,
        model,
        img_save_dir,
        n_images,
        generator,
        prompts,
        n_inference_steps=10,
        img_dim=(256,256), 
    ):
        """
        Given a list of prompts, generate a total number of n_images images. Images are saved to img_save_dir/prompt_with_spaces_replaced_by_underscores.

        Args:
            model : text-to-image model used to generate the images
            img_save_dir (str) : where to save the generated images
            prompts (list(str)) : lsit of prompts
            n_inference_steps (int) : number of denoising steps to use for image generation
            img_dim (2-tuple(int)) : size of images to generate
        """

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
            self.generate_images(
                model=model,
                img_save_dir=img_save_folder,
                prompt=prompt,
                n_images=n,
                generator=generator,
                n_inference_steps=n_inference_steps,
                img_dim=img_dim,
            )

    def generate_cat_images(
        self,
        model,
        img_save_dir,
        n_images,
        generator,
        n_inference_steps=10,
        img_dim=(256,256),
    ):
        self.generate_images_from_prompt_list(
            model=model,
            img_save_dir=img_save_dir,
            n_images=n_images,
            generator=generator,
            prompts=self.TASK_TO_PROMPTS["cat"],
            n_inference_steps=n_inference_steps,
            img_dim=img_dim,
        )
     