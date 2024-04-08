
import os
import numpy as np
import torch
import torchvision
import argparse
import math

import datetime
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel


def visualize_similar_samples_sd_intermediate_features(images, save_dir="similarity_visualizations"):

    # if inputs are image paths, read them
    if isinstance(images, list):
        images = torch.stack([torchvision.io.read_image(image_path) for image_path in images])

    # get dictionary of intermediate_features    
    intermediate_features = _compute_sd_intermediate_latents(images=images)

    # compute distances and save visualizations for each intermediate features
    for key in intermediate_features:
        name = key.replace('.', '_')
        save_folder = os.path.join(save_dir, name)
        os.makedirs(save_folder)
        features = intermediate_features[key]
        dists = _compute_l2_distance(features, features)
        save_visualizations(images=images, distances=dists, save_dir=save_folder)
        print(f"saved {save_folder}")
        breakpoint()

def visualize_similar_samples(images, similarity_type="sd_latent_l2", save_dir="similarity_visualizations"):
    """
    Args:
        images (tensor or list(str)) : images to visualize similarity among
            Given as tensor (n_images, D, W, H) or as a list of paths to saved images
        similarity_type (str) : type of similarity metric to use
        save_dir (str) : path to directory to save visualizations
    """
    SIMILARITY_FUNCTIONS = {
        "lpips" : _compute_lpips_distance,
        "sd_latent_l2" : _compute_sd_latent_l2_distance,
    }

    similarity_fn = SIMILARITY_FUNCTIONS[similarity_type]

    # if inputs are image paths, read them
    if isinstance(images, list):
        images = torch.stack([torchvision.io.read_image(image_path) for image_path in images])
    
    distances = similarity_fn(images)

    save_visualizations(images=images, distances=distances, save_dir=save_dir)


# def visualize_similar_samples(images, similarity_type="sd_latent_l2", save_dir="similarity_visualizations"):
#     """
#     Args:
#         images (tensor or list(str)) : images to visualize similarity among
#             Given as tensor (n_images, D, W, H) or as a list of paths to saved images
#         similarity_type (str) : type of similarity metric to use
#         save_dir (str) : path to directory to save visualizations
#     """
#     SIMILARITY_FUNCTIONS = {
#         "lpips" : _compute_lpips_distance,
#         "sd_latent_l2" : _compute_sd_latent_l2_distance,
#     }

#     similarity_fn = SIMILARITY_FUNCTIONS[similarity_type]

#     # if inputs are image paths, read them
#     if isinstance(images, list):
#         images = torch.stack([torchvision.io.read_image(image_path) for image_path in images])

#     # get grid cell size and scaled image size
#     n_images = images.shape[0]
#     n_rows = int(np.ceil(np.sqrt(n_images)))
#     grid_size = (n_rows, n_rows)
#     output_img_size = (1024, 1024)
#     grid_cell_size = (int(output_img_size[0] / n_rows), int(output_img_size[0] / n_rows))
#     image_width = int(0.9*grid_cell_size[0])
#     image_height = int(0.9*grid_cell_size[1])

#     distances = similarity_fn(images)

#     for i, dist in enumerate(distances):
#         # order indices from most to least similar
#         most_to_least_similar = torch.argsort(dist)

#         # create canvas
#         grid_width = grid_size[1] * grid_cell_size[0]
#         grid_height = grid_size[0] * grid_cell_size[1]
#         grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
#         draw = ImageDraw.Draw(grid_image)
#         font = ImageFont.truetype("arial.ttf", 12)

#         for j, img_idx in enumerate(most_to_least_similar):
#             # place image on grid
#             img = torchvision.transforms.functional.to_pil_image(images[img_idx])
#             x_offset = (j % grid_size[1]) * grid_cell_size[0]
#             y_offset = (j // grid_size[0]) * grid_cell_size[1]
#             grid_image.paste(img.resize((image_width, image_height)), (x_offset, y_offset))

#             # add score text
#             text = f"{dist[img_idx]}"
#             # text_width, text_height = draw.textsize(text, font=font)
#             text_position = (x_offset + 5, y_offset + image_height + 5)
#             draw.text(text_position, text, fill='black', font=font)
#         grid_image.save(os.path.join(save_dir, f"{i}_similarities.jpg"))

# def visualize_similar_samples(images, save_dir="similarity_visualizations"):
#     """
#     Args:
#         images (tensor or list(str)) : images to visualize similarity among
#             Given as tensor (n_images, D, W, H) or as a list of paths to saved images
#         save_dir (str) : path to directory to save visualizations
#     """
#     import lpips
#     loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
#     loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

#     # if inputs are image paths, read them
#     if isinstance(images, list):
#         images = torch.stack([torchvision.io.read_image(image_path) for image_path in images])

#     n_images = images.shape[0]
#     # image_width = images[0].shape[1]
#     # image_height = images[0].shape[2]
#     n_rows = int(np.ceil(np.sqrt(n_images)))
#     grid_size = (n_rows, n_rows)
#     output_img_size = (1024, 1024)
#     grid_cell_size = (int(output_img_size[0] / n_rows), int(output_img_size[0] / n_rows))
#     image_width = int(0.9*grid_cell_size[0])
#     image_height = int(0.9*grid_cell_size[1])

#     for i, im1 in enumerate(images):
#         # get distances to each image
#         dists = []
#         for im2 in images:
#             dists.append(loss_fn_alex(im1, im2))
#         dists = torch.tensor(dists)
#         most_to_least_similar = torch.argsort(dists)

#         # create canvas
#         grid_width = grid_size[1] * grid_cell_size[0]
#         grid_height = grid_size[0] * grid_cell_size[1]
#         grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
#         draw = ImageDraw.Draw(grid_image)
#         font = ImageFont.truetype("arial.ttf", 14)

#         for j, img_idx in enumerate(most_to_least_similar):
#             # place image on grid
#             img = torchvision.transforms.functional.to_pil_image(images[img_idx])
#             x_offset = (j % grid_size[1]) * grid_cell_size[0]
#             y_offset = (j // grid_size[0]) * grid_cell_size[1]
#             grid_image.paste(img.resize((image_width, image_height)), (x_offset, y_offset))

#             # add score text
#             text = f"{dists[img_idx]}"
#             # text_width, text_height = draw.textsize(text, font=font)
#             text_position = (x_offset + 5, y_offset + image_height + 5)
#             draw.text(text_position, text, fill='black', font=font)
#         grid_image.save(os.path.join(save_dir, f"{i}_similarities.jpg"))

def save_visualizations(images, distances, save_dir):
    
    # get grid cell size and scaled image size
    n_images = distances.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_images)))
    grid_size = (n_rows, n_rows)
    output_img_size = (1024, 1024)
    grid_cell_size = (int(output_img_size[0] / n_rows), int(output_img_size[0] / n_rows))
    image_width = int(0.9*grid_cell_size[0])
    image_height = int(0.9*grid_cell_size[1])


    for i, dist in enumerate(distances):
        # order indices from most to least similar
        most_to_least_similar = torch.argsort(dist)

        # create canvas
        grid_width = grid_size[1] * grid_cell_size[0]
        grid_height = grid_size[0] * grid_cell_size[1]
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.truetype("arial.ttf", 12)

        for j, img_idx in enumerate(most_to_least_similar):
            # place image on grid
            img = torchvision.transforms.functional.to_pil_image(images[img_idx])
            x_offset = (j % grid_size[1]) * grid_cell_size[0]
            y_offset = (j // grid_size[0]) * grid_cell_size[1]
            grid_image.paste(img.resize((image_width, image_height)), (x_offset, y_offset))

            # add score text
            text = f"{dist[img_idx]}"
            # text_width, text_height = draw.textsize(text, font=font)
            text_position = (x_offset + 5, y_offset + image_height + 5)
            draw.text(text_position, text, fill='black', font=font)
        grid_image.save(os.path.join(save_dir, f"{i}_similarities.jpg"))


##################################################
# DISTANCE METRICS #
##################################################

def _compute_l2_distance(features1, features2): # TODO - outdated?
    """
    Computes L2 distance between each feature in features1 to all features in features2
    Args:
        features1 (Tensor) : features or images
        features2 (Tensor) : features or images to compare features in feature1 to
    Returns:
        distances (tensor) : each row is a distance from sample i to all other samples including itself
    """

    distances = []
    for f in features1:
        dists = (features2 - f).pow(2).mean(dim=tuple(np.arange(1, features1.dim())))
        distances.append(dists.tolist())

    return torch.tensor(distances)

def _compute_lpips_distance(images):
    """
    Compute LPIPS loss
    Args:
        images (tensor) : (N, D, W, H)
    Returns:
        distances (tensor) : distance between each of the input images
            [[dist(im0, im0), dist(im0, im1), ... dist(im0, imn)], [dist(im1, im0), dist(im1, im1)...]...]
    """
    import lpips

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    distances = torch.zeros((images.shape[0], images.shape[0]))
    for i, im1 in enumerate(images):    
        for j, im2 in enumerate(images):
            if not i == j:
                distances[i,j] = loss_fn_alex(im1, im2)

    return distances

def _compute_sd_latent_l2_distance(images):
    """
    Compute L2 loss on SD encoded latents
    Args:
        images (tensor) : (N, D, W, H)
    Returns:
        distances (tensor) : distance between each of the input images
            [[dist(im0, im0), dist(im0, im1), ... dist(im0, imn)], [dist(im1, im0), dist(im1, im1)...]...]
    """
    # initialize model
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    pretrained_revision = "main"
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path, 
        revision=pretrained_revision
    ).to("cuda")

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # run input images through SD image encoder
    encoder = pipeline.vae.encoder
    latents = encoder(images[:2].float().to("cuda")) 

    return _compute_l2_distance(latents, latents)

def _compute_sd_intermediate_latents(images):
    intermediate_features = {}
    def get_activation(name):
        def hook(model, input, output):
            intermediate_features[name] = output.detach()
        return hook

    # initialize model
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    pretrained_revision = "main"
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path, 
        revision=pretrained_revision
    ).to("cuda")

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # run input images through SD image encoder
    encoder = pipeline.vae.encoder
    for name, layer in encoder.named_modules():
        layer.register_forward_hook(get_activation(name))

    latents = encoder(images.float().to("cuda")) 
    return intermediate_features


def main(args):
    images = [os.path.join(args.img_dir, img_file) for img_file in os.listdir(args.img_dir)]
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    os.makedirs(save_dir)
    visualize_similar_samples_sd_intermediate_features(images=images, save_dir=save_dir)
    # visualize_similar_samples(images=images, save_dir=save_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, help="directory where source images are", default="/home/hayano/similarity_test_images")
    parser.add_argument("--save-dir", type=str, help="directory to save visualizations", default="/home/hayano/similarity_visualizations")
    
    args = parser.parse_args()
    main(args)


# # img_dir = "/home/hayano/similarity_test_images"
# img_dir = "/home/hayano/test"

# images = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)]
# save_dir = "/home/hayano/similarity_visualizations"
# os.makedirs(save_dir, exist_ok=True)
# save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
# os.makedirs(save_dir)

# visualize_similar_samples(images=images, save_dir=save_dir)