from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
import csv
import pandas as pd
import random
import os
import time
import itertools
from PIL import Image
import random
import numpy as np
import argparse
from datetime import datetime
import shutil

from rl4dgm.utils.query_generator import PreferenceQueryGenerator
from rl4dgm.utils.generate_images import ImageGenerator

NUM_ITERS_TO_RUN = 3
NUM_INFERENCE_STEPS = 10

device = "cuda"

def generate_images(
    model,
    img_save_dir, 
    prompt, 
    n_images,
    n_inference_steps=10,
    img_dim=(256,256),
    seed=None,
):
    if seed is None:
        seed = torch.random.seed()
    generator = torch.manual_seed(seed)

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

def create_pickapic_dataframe():
    return pd.DataFrame(
        columns=[
            "are_different", "best_image_uid", "caption", "created_at", "has_label",
            "image_0_uid", "image_0_url", "image_1_uid", "image_1_url",
            "jpg_0", "jpg_1",
            "label_0", "label_1", "model_0", "model_1",
            "ranking_id", "user_id", "num_example_per_prompt", #"__index_level_0__",
        ],
    )

def add_to_pickapic_dataframe(preference_df, prompt, labels, image_paths):

    are_different = not image_paths[0] == image_paths[1]
    best_image_uid = ""
    created_at = datetime.now()
    has_label = True
    image_0_uid = "0"
    image_0_url = ""
    image_1_uid = "1"
    image_1_url = ""
    with open(image_paths[0], "rb") as img0:
        jpg_0 = img0.read()

    with open(image_paths[1], "rb") as img1:
        jpg_1 = img1.read()
    model0 = ""
    model1 = ""
    ranking_id = 0
    user_id = 0
    num_example_per_prompt = 1

    label0, label1 = labels

    preference_df.loc[len(preference_df.index)] = [
        are_different, best_image_uid, prompt, created_at, has_label,
        image_0_uid, image_0_url, image_1_uid, image_1_url,
        jpg_0, jpg_1,
        label0, label1, model0, model1,
        ranking_id, user_id, num_example_per_prompt,
    ]

    return preference_df


def generate_dummy_icecream_dataset(model, data_save_dir, n_images, n_queries, datafile_name="dummy_icecream.parquet", seed=None):
    
    #### Generate Images ####
    prompts = [
        "Premium soft serve ice cream",
        "The world's most expensive soft serve ice cream",
        "The world's most expensive soft serve ice cream topped with edible gold and jewerly-like fruits",
        "The world's most expensive soft serve ice cream topped with edible gold and a tiara decorated with jewerly-like fruits"
    ]

    img_dirs = []
    for prompt in prompts:
        img_save_dir = os.path.join(data_save_dir, "image_data", prompt.replace(" ", "_"))
        img_dirs.append(img_save_dir)
        generate_images(model=model, img_save_dir=img_save_dir, prompt=prompt, n_images=25, seed=seed)

    #### Generate Preference Data ####
    # initialize dataframe
    preference_df = create_pickapic_dataframe()

    # generate random queries
    query_generator = PreferenceQueryGenerator()
    queries, img_paths = query_generator.generate_queries(image_directories=img_dirs, query_algorithm="random", n_queries=100)
    for query in queries:
        im0 = img_paths[query[0]]
        im1 = img_paths[query[1]]

        label0, label1 = preference_from_ranked_prompts(prompts=prompts, img_paths=[im0, im1])

        preference_df = add_to_pickapic_dataframe(
            preference_df=preference_df, 
            prompt="Premium soft serve ice cream",
            labels=(label0, label1), 
            image_paths=(im0, im1),
        )
    
    preference_df.to_parquet(os.path.join(data_save_dir, datafile_name))
    print("Saved to: ", os.path.join(data_save_dir, datafile_name))

    print("removing saved images...")
    shutil.rmtree(os.path.join(data_save_dir, "image_data"))

def generate_dummy_cat_dataset(model, data_save_dir, n_images, n_queries, datafile_name="dummy_cat.parquet", seed=None):

    #### Generate Images ####
    prompts = [
        "A cute black cat with round eyes",
        "A demonic black cat with sharp eyes",
        "A cute white cat with round eyes",
        "A demonic white cat with sharp eyes",
    ]

    img_dirs = []
    for prompt in prompts:
        img_save_dir = os.path.join(data_save_dir, "image_data", prompt.replace(" ", "_"))
        img_dirs.append(img_save_dir)
        generate_images(model=model, img_save_dir=img_save_dir, prompt=prompt, n_images=n_images, seed=seed)
    
    #### Generate Preference Data ####
    # initialize dataframe
    preference_df = create_pickapic_dataframe()

    # generate random queries
    query_generator = PreferenceQueryGenerator()
    queries, img_paths = query_generator.generate_queries(images=img_dirs, query_algorithm="random", n_queries=100)
    for query in queries:
        im0 = img_paths[query[0]]
        im1 = img_paths[query[1]]
        label0, label1 = preference_from_keyphrases(keyphrases=["black", "cute"], img_paths=[im0, im1])

        preference_df = add_to_pickapic_dataframe(
            preference_df=preference_df, 
            prompt="A cute cat",
            labels=(label0, label1), 
            image_paths=(im0, im1),
        )
    
    preference_df.to_parquet(os.path.join(data_save_dir, datafile_name))
    print("Saved to: ", os.path.join(data_save_dir, datafile_name))

    print("removing saved images...")
    shutil.rmtree(os.path.join(data_save_dir, "image_data"))

def generate_dummy_cat_random_ranked_dataset(model, data_save_dir, n_images, n_queries, datafile_name="dummy_ranked.parquet", seed=None):

    image_generator = ImageGenerator()
    if seed is None:
        seed = torch.random.seed()
    generator = torch.manual_seed(seed)

    #### Generate Images ####
    prompts = [
        "A cute black cat with round eyes",
        "A demonic black cat with sharp eyes",
        "A cute white cat with round eyes",
        "A demonic white cat with sharp eyes",
    ]

    img_save_dir = os.path.join(data_save_dir, "image_data")

    image_generator.generate_images_from_prompt_list(
        model=model,
        generator=generator,
        n_images=100,
        img_save_dir=img_save_dir,
        prompts=prompts,
        separate_by_prompt=True,           
    )

    #### Generate Preference Data ####
    preference_df = create_pickapic_dataframe()

    # generate queries
    query_generator = PreferenceQueryGenerator()
    queries, img_paths = query_generator.generate_queries(images=[img_save_dir], query_algorithm="ordered", n_queries=100)
    for i, query in enumerate(queries):
        im0 = img_paths[query[0]]
        im1 = img_paths[query[1]]
        label0, label1 = preference_from_image_order(img_paths=[im0, im1])
        preference_df = add_to_pickapic_dataframe(
            preference_df=preference_df, 
            prompt="a cute cat",
            labels=(label0, label1), 
            image_paths=(im0, im1),
        )
    preference_df.to_parquet(os.path.join(data_save_dir, datafile_name))
    print("Saved to: ", os.path.join(data_save_dir, datafile_name))

    # print("removing saved images...")
    # shutil.rmtree(os.path.join(data_save_dir, "image_data"))


# def generate_dummy_cat_random_ranked_dataset(model, data_save_dir, n_images, n_queries, datafile_name="dummy_ranked.parquet", seed=None):
    
#     #### Generate Images ####
#     prompts = [
#         "A cute black cat with round eyes",
#         "A demonic black cat with sharp eyes",
#         "A cute white cat with round eyes",
#         "A demonic white cat with sharp eyes",
#     ]

#     img_dirs = []
#     for prompt in prompts:
#         img_save_dir = os.path.join(data_save_dir, "image_data", prompt.replace(" ", "_"))
#         img_dirs.append(img_save_dir)
#         generate_images(model=model, img_save_dir=img_save_dir, prompt=prompt, n_images=n_images, seed=seed)

#     # prompt = "a cute cat"
#     # img_save_dir = os.path.join(data_save_dir, "image_data")
#     # generate_images(img_save_dir=img_save_dir, prompt=prompt, n_images=100, seed=seed)

#     #### Generate Preference Data ####
#     preference_df = create_pickapic_dataframe()

#     # generate queries
#     query_generator = QueryGenerator()
#     queries, img_paths = query_generator.generate_queries(images=img_dirs, query_algorithm="ordered", n_queries=100)
#     for i, query in enumerate(queries):
#         im0 = img_paths[query[0]]
#         im1 = img_paths[query[1]]
#         label0, label1 = preference_from_image_order(img_paths=[im0, im1])
#         preference_df = add_to_pickapic_dataframe(
#             preference_df=preference_df, 
#             prompt=prompt,
#             labels=(label0, label1), 
#             image_paths=(im0, im1),
#         )
#     preference_df.to_parquet(os.path.join(data_save_dir, datafile_name))
#     print("Saved to: ", os.path.join(data_save_dir, datafile_name))

#     print("removing saved images...")
#     shutil.rmtree(os.path.join(data_save_dir, "image_data"))

def preference_from_ranked_prompts(prompts, img_paths, **kwargs):
    """
    Generate labels for images given a list of prompts ordered from low to high scores, and a list of image paths containing one of the prompts

    Args:
        prompts (list(str)) : list of prompts, ordered from one that correspond to low score to high score
        img_paths (list(str)) : list of paths to images in the query. Each image path should contain one of the prompts with spaces replaced with underscores
    
    Returns: 
        labels (tuple(float)) : label0, label1 where preferred image is labeled with 1 and non-preferred image is labeld with 0. If tie, both images are labeled 0.5
    """

    assert len(img_paths) == 2, f"Expected 2 image paths. Got {len(img_paths)}"

    img_dirs = []
    for prompt in prompts:
        img_dirs.append(prompt.replace(" ", "_"))

    im0, im1 = img_paths

    im0_score = np.where(np.array([img_dir in im0 for img_dir in img_dirs]))[0].max()
    im1_score = np.where(np.array([img_dir in im1 for img_dir in img_dirs]))[0].max()

    if im0_score > im1_score:
        label0 = 1
        label1 = 0
    elif im1_score > im0_score:
        label0 = 0
        label1 = 1
    else:
        label0 = 0.5
        label1 = 0.5
    
    return label0, label1

def preference_from_keyphrases(keyphrases, img_paths, **kwargs):
    """
    Generate labels for images given a preferred keyword or keyphrase, and a list of image paths containing the prompt.

    Args:
        keyphrases (list(str)) : keyphrases to look for in a query image in order of priority. Image with this keyphrase contained in the path is preferred.
            If both images contain / do not contain the first keyphrase, the one containing the next keyphrase is preferred.
            If both images contain (do not contain) all (any) of the keyphrases, labels of 0.5 are given. 
            Spaces in the keyphrase are replaced by underscores. Case insensitive.
        img_paths (list(str)) : list of paths to images in the query

    Returns:
        labels (tuple(float)) : label0, label1 where preferred image is labeled with 1 and non-preferred image is labeld with 0. If tie, both images are labeled 0.5
    """

    assert len(img_paths) == 2, f"Expected 2 image paths. Got {len(img_paths)}"
    im0, im1 = img_paths

    for keyphrase in keyphrases:
        keyphrase = keyphrase.replace(" ", "_")

        keyin0 = keyphrase.lower() in im0.lower()
        keyin1 = keyphrase.lower() in im1.lower()

        if keyin0 and not keyin1:
            label0 = 1
            label1 = 0
            return label0, label1

        elif keyin1 and not keyin0:
            label0 = 0
            label1 = 1
            return label0, label1

    # both images contain all or none of the keyphrases -> no preference
    label0 = 0.5
    label1 = 0.5
    return label0, label1

def preference_from_image_order(img_paths, **kwargs):
    """
    Generate labels for images, prefering the first one over the second one
    """
    assert len(img_paths) == 2, f"Expected 2 image paths. Got {len(img_paths)}"
    label0 = 1
    label1 = 0
    return label0, label1

def main(args):

    model = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(device)

    TYPE_TO_DATASET = {
        "cat" : generate_dummy_cat_dataset,
        "icecream" : generate_dummy_icecream_dataset,
        "cat_ordered" : generate_dummy_cat_random_ranked_dataset,
    }

    assert args.type in TYPE_TO_DATASET.keys(), f"Type must be one of {TYPE_TO_DATASET.keys()}. Got {args.type}"

    dummy_dataset_generator = TYPE_TO_DATASET[args.type]

    if args.all_sets is None:
        dummy_dataset_generator(
            model=model,
            data_save_dir=args.save_dir,
            n_images=args.n_images,
            n_queries=args.n_queries,
            datafile_name=args.parquet_filename,
            seed=args.seed,
        )

    else:
        datafile_names = [f"{args.all_sets}_train.parquet", f"{args.all_sets}_validation.parquet", f"{args.all_sets}_test.parquet"]
        for datafile_name in datafile_names:
            dummy_dataset_generator(
                model=model,
                data_save_dir=args.save_dir,
                n_images=args.n_images,
                n_queries=args.n_queries,
                datafile_name=datafile_name,
                seed=args.seed,
            )
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="dummy_dataset")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--type", type=str, help=f"Which type of dummy dataset to generate")
    parser.add_argument("--parquet-filename", type=str, default="dummy_dataset.parquet")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-sets", type=str, help="filename prefix to generate train, test, and validation datasets. If provided, ignores --parquet-filename argument")
    args = parser.parse_args()
    main(args)
