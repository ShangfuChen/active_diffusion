
import os
import math

import numpy as np
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse

from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

from rl4dgm.utils.reward_processor import RewardProcessor

# AI agents
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

# LAION Aesthetics evaluator
from ddpo.aesthetic_scorer import AestheticScorer
def aesthetic_score_fn(images, prompts, **kwargs):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
    else:
        images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        images = torch.tensor(images, dtype=torch.uint8)
    scores = kwargs["scorer"](images)
    return scores, {}

# PickScore
def pickscore_score_fn(images, prompts, **kwargs):
    """
    Args: processor, model, images, prompts, device
    """
    if isinstance(images, torch.Tensor):
        imgs = []
        toPIL = torchvision.transforms.ToPILImage()
        for im in images:
            imgs.append(toPIL(im))    
    _, score = calc_probs(kwargs["processor"], kwargs["model"], imgs, prompts, device)
    return score, {}




#####################################################################################
MODELS = ["laion", "pickscore"]
FEATURIZERS = ["sd"]

def initialize_model(model_type):
    """
    Return evaluator model and input arguments EXCLUDING imges and prompts
    """
    assert model_type in MODELS, f"model_type must be one of {MODELS}. Got {model_type}"
    if model_type == "laion":
        score_fn = aesthetic_score_fn
        score_fn_inputs = {
            "scorer" : AestheticScorer(dtype=torch.float32).cuda(),
        }
    elif model_type == "pickscore":
        score_fn = pickscore_score_fn
        score_fn_inputs = {
            "processor" : AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
            "model" : AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device),
            "device" : "cuda",
        }

    return score_fn, score_fn_inputs

def initialize_featurizer(featurizer_type):
    """
    Get featurizer function 
    """
    assert featurizer_type in FEATURIZERS, f"featurizer_type must be one of {FEATURIZERS}. Got {featurizer_type}."
    if featurizer_type == "sd":
        # SD encoder
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
        featurizer_fn = pipeline.vae.encoder

    return featurizer_fn

def featurize(images, featurizer_fn, featurizer_type):
    assert featurizer_type in FEATURIZERS, f"featurizer_type must be one of {FEATURIZERS}. Got {featurizer_type}."
    if featurizer_type == "sd":
        latents = featurizer_fn(images.float().to("cuda")).cpu()
    
    return latents

def extract_images(input_dir, max_images_per_epoch=None,):
    image_paths = [] # image paths
    images = [] # image tensors
    for epoch in os.listdir(input_dir):
        # print("storing images for epoch", epoch)
        n_saved_images = 0
        for img in os.listdir(os.path.join(input_dir, epoch)):
            if max_images_per_epoch is not None and n_saved_images >= max_images_per_epoch:
                break
            filepath = os.path.join(input_dir, epoch, img)
            image_paths.append(filepath)
            images.append(torchvision.io.read_image(filepath))
            n_saved_images += 1

    return img_paths, torch.stack(images)

def get_reward_labels(images, ai_score_fn, ai_score_fn_args, human_score_fn, human_score_fn_args, max_num_images=960, ):
    max_n_images = 960
    n_batches = math.ceil(images.shape[0] / max_n_images)
    ai_rewards = []
    human_rewards = []

    for i in range(n_batches):
        print("batch", i)
        start_idx = i * max_n_images
        end_idx = images.shape[0] if i == n_batches - 1 else (i+1) * max_n_images
        ai_scores, _ = ai_score_fn(images=images[start_idx:end_idx], prompts="a cute cat", **ai_score_fn_args)
        ai_rewards += ai_scores.cpu().tolist()
        human_scores, _ = human_score_fn(images=images[start_idx:end_idx], prompts="a cute cat", **human_score_fn_args)
        human_rewards += human_scores

    return ai_rewards, human_rewards

def main(args):
    # initialize according to human and AI agents
    ai_evaluator, ai_evaluator_args = initialize_model(model_type=args.ai)
    human_evaluator, human_evaluator_args = initialize_model(model_type=args.human)

    # read images from input directory
    img_paths, imgs = extract_images(input_dir=args.input_dir)

    # get AI and human rewards
    ai_rewards, human_rewards = get_reward_labels() # TODO start here
    # collect data

    # plot

    pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input-dir", type=str)
#     parser.add_argument("--ai", type=str, default="laion")
#     parser.add_argument("--human", type=str, default="pickscore")
#     parser.add_argument("--featurizer", type=str, default="sd")
#     parser.add_argument("--save-dir", type=str, default="/home/ayanoh/ai_correction_experiments")
#     parser.add_argument("--name", type=str)


#############################################################
# Initialization
#############################################################
input_dir = "/home/ayanoh/all_aesthetic"
device = "cuda"
df = pd.DataFrame(
    columns=[
        "img_paths", 
        "human_rewards",
        "ai_rewards",
        "errors", # R_H - R_AI
    ],
)
reward_processor = RewardProcessor()

# dummy human and AI evaluator models
aesthetics_evaluator = AestheticScorer(dtype=torch.float32).cuda()
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# model for feature extraction
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
encoder = pipeline.vae.encoder

#############################################################
# Extract images
#############################################################
img_paths = [] # image paths
imgs = [] # image tensors
n_images_per_epoch = 256 # number of images to save from each epoch
for epoch in os.listdir(input_dir):
    # print("storing images for epoch", epoch)
    n_saved_images = 0
    for img in os.listdir(os.path.join(input_dir, epoch)):
        if n_saved_images >= n_images_per_epoch:
            break
        filepath = os.path.join(input_dir, epoch, img)
        img_paths.append(filepath)
        imgs.append(torchvision.io.read_image(filepath))
        n_saved_images += 1
    # print("done")
print("finished storing images for all epochs")
    

#############################################################
# Get human and AI reward labels
#############################################################
# pass images in incrementally if total number of images is too large for single batch
max_n_images = 960
imgs = torch.stack(imgs)
n_batches = math.ceil(imgs.shape[0] / max_n_images)
ai_rewards = []
human_rewards = []

for i in range(n_batches):
    print("batch", i)
    start_idx = i * max_n_images
    end_idx = imgs.shape[0] if i == n_batches - 1 else (i+1) * max_n_images
    ai_scores, _ = aesthetic_score_fn(images=imgs[start_idx:end_idx], prompts="a cute cat", scorer=aesthetics_evaluator)
    ai_rewards += ai_scores.cpu().tolist()
    human_scores, _ = pickscore_score_fn(processor=pickscore_processor, model=pickscore_model, images=imgs[start_idx:end_idx], prompts="a cute cat", device=device)
    human_rewards += human_scores

df["img_paths"] = img_paths
df["human_rewards"] = human_rewards
df["ai_rewards"] = ai_rewards
df["errors"] = np.array(human_rewards) - np.array(ai_rewards) 
print("human and ai dataframe is ready")

#############################################################
# Record relationship between human dataset size and error
#############################################################
n_samples = imgs.shape[0]
chunk_size = 10
n_chunks = math.ceil(n_samples / chunk_size)
print("computing first batch of features")
human_dataset_features = encoder(imgs[:chunk_size].float().to("cuda")).cpu()
print("first batch of features added to human dataset")

# values to record for plotting
corrected_ai_feedback_errors = []
corrected_ai_feedback_percent_errors = []
ai_feedback_errors = []
ai_feedback_percent_errors = []
avg_min_distances = []
min_ai_rewards_this_batch = []
min_ai_rewards_overall = []
max_ai_rewards_this_batch = []
max_ai_rewards_overall = []
min_human_rewards_this_batch = []
min_human_rewards_overall = []
max_human_rewards_this_batch = []
max_human_rewards_overall = []

n_human_feedback = []
for chunk in range(1, n_chunks): 
    if chunk % 10 == 0: 
        print(f"chunk {chunk} / {n_chunks}")
    start_idx = chunk * chunk_size
    end_idx = n_samples if chunk == n_chunks - 1 else start_idx + chunk_size

    # extract features
    latents = encoder(imgs[start_idx:end_idx].float().to("cuda")).cpu()

    # compute similarity and AI feedback correction
    min_distances, most_similar_indices = reward_processor._get_most_similar_l2(features1=latents, features2=human_dataset_features)
    avg_min_distances.append(min_distances.mean())
    # dists = _compute_l2_distance(features1=latents, features2=human_dataset_features)

    # get corrected AI feedback 
    ai_rewards_for_this_batch = np.array(df["ai_rewards"][start_idx:end_idx])
    errors_for_this_batch = np.array(df["errors"])[most_similar_indices]
    ai_rewards_corrected = ai_rewards_for_this_batch + errors_for_this_batch

    # compare corrected AI feedback and human feedback
    human_rewards_for_this_batch = np.array(df["human_rewards"][start_idx:end_idx])
    error_corrected_ai_and_human = human_rewards_for_this_batch - ai_rewards_corrected

    # log for plotting
    corrected_ai_feedback_errors.append(error_corrected_ai_and_human.mean())
    human_rewards_range = max(df["human_rewards"][:end_idx]) - min(df["human_rewards"][:end_idx]) # range of human rewards so far
    percent_error = (error_corrected_ai_and_human / human_rewards_range).mean()
    corrected_ai_feedback_percent_errors.append(percent_error.tolist())

    error_raw_ai_and_human = human_rewards_for_this_batch - ai_rewards_for_this_batch
    ai_feedback_errors.append(error_raw_ai_and_human.mean())
    ai_feedback_percent_errors.append((error_raw_ai_and_human / human_rewards_range).mean().tolist())
    
    min_ai_rewards_this_batch.append(min(ai_rewards_for_this_batch))
    min_ai_rewards_overall.append(min(df["ai_rewards"][:end_idx]))
    max_ai_rewards_this_batch.append(max(ai_rewards_for_this_batch))
    max_ai_rewards_overall.append(max(df["ai_rewards"][:end_idx]))
    min_human_rewards_this_batch.append(min(human_rewards_for_this_batch))
    min_human_rewards_overall.append(min(df["human_rewards"][:end_idx]))
    max_human_rewards_this_batch.append(max(human_rewards_for_this_batch))
    max_human_rewards_overall.append(max(df["human_rewards"][:end_idx]))

    # add this current batch to the human dataset
    human_dataset_features = torch.cat([human_dataset_features, latents])
    n_human_feedback.append(start_idx)
    
#############################################################
# Plot results
#############################################################
save_dir = os.path.join("/home/ayanoh/ai_correction_experiments", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
os.makedirs(save_dir, exist_ok=True)

# Error between corrected AI feedback and human feedback
print("saving AI-human error (raw)")
plt.plot(n_human_feedback, ai_feedback_errors)
plt.title("Error between Raw AI Feedback and Human Feedback")
plt.xlabel("N Human Feedback")
plt.ylabel("Score Error")
plt.savefig(os.path.join(save_dir, "raw_ai_human_error.jpg"))
plt.clf()

# Percent Error between corrected AI feedback and human feedback
print("saving AI-human percent error")
plt.plot(n_human_feedback, ai_feedback_percent_errors)
plt.title("Percent Error between Raw AI Feedback and Human Feedback")
plt.xlabel("N Human Feedback")
plt.ylabel("Score Error")
plt.savefig(os.path.join(save_dir, "raw_ai_human_percent_error.jpg"))
plt.clf()


# Error between corrected AI feedback and human feedback
print("saving AI-human error")
plt.plot(n_human_feedback, corrected_ai_feedback_errors)
plt.title("Error between Corrected AI Feedback and Human Feedback")
plt.xlabel("N Human Feedback")
plt.ylabel("Score Error")
plt.savefig(os.path.join(save_dir, "corrected_ai_human_error.jpg"))
plt.clf()

# Percent Error between corrected AI feedback and human feedback
print("saving AI-human percent error")
plt.plot(n_human_feedback, corrected_ai_feedback_percent_errors)
plt.title("Percent Error between Corrected AI Feedback and Human Feedback")
plt.xlabel("N Human Feedback")
plt.ylabel("Score Error")
plt.savefig(os.path.join(save_dir, "corrected_ai_human_percent_error.jpg"))
plt.clf()

# Mean minimum distance between current batch samples and samples in human dataset
print("saving min and max rewards")
plt.plot(n_human_feedback, avg_min_distances)
plt.title("Avg Min Distance to Human Dataset")
plt.xlabel("N Human Feedback")
plt.ylabel("Average Minimum Distance")
plt.savefig(os.path.join(save_dir, "avg_min_dist_to_human_dataset.jpg"))
plt.clf()

# Minimum and maximum AI and human feedback for each batch and overall
plt.fill_between(n_human_feedback, min_ai_rewards_this_batch, max_ai_rewards_this_batch, alpha=0.2)
plt.fill_between(n_human_feedback, min_ai_rewards_overall, max_ai_rewards_overall, alpha=0.2)
plt.fill_between(n_human_feedback, min_human_rewards_this_batch, max_human_rewards_this_batch, alpha=0.2)
plt.fill_between(n_human_feedback, min_human_rewards_overall, max_human_rewards_overall, alpha=0.2)
plt.title("AI and Human Feedback Range")
plt.xlabel("N Human Feedback")
plt.ylabel("Feedback Range")
plt.legend(["AI (this batch)", "AI (overall)", "Human (this batch)", "Human (overall)"])
plt.savefig(os.path.join(save_dir, "feedback_range.jpg"))
plt.clf()

print("Done saving figures")