
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
    _, score = calc_probs(kwargs["processor"], kwargs["model"], imgs, prompts, kwargs["device"])
    return score, {}




#####################################################################################
MODELS = ["laion", "pickscore"]
FEATURIZERS = ["sd"]

def initialize_model(model_type, device):
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
            "device" : device,
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

def extract_images(input_dir=None, image_paths=None, max_images_per_epoch=None,):
    assert sum([input_dir is not None, image_paths is not None]) == 1, "Either input_dir or image_paths should be provided (but not both)"
    if input_dir is not None:
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
    else:
        images = []
        for filepath in image_paths:
            n_saved_images = 0
            if max_images_per_epoch is not None and n_saved_images >= max_images_per_epoch:
                break
            images.append(torchvision.io.read_image(filepath))
            n_saved_images += 1
    
    return image_paths, torch.stack(images)

def get_reward_labels(
    images, 
    ai_score_fn, 
    ai_score_fn_args, 
    human_score_fn, 
    human_score_fn_args, 
    max_num_images=960, 
):
    n_batches = math.ceil(images.shape[0] / max_num_images)
    ai_rewards = []
    human_rewards = []

    # if there are too many images to process in a single batch, split 
    for i in range(n_batches):
        print("batch", i)
        start_idx = i * max_num_images
        end_idx = images.shape[0] if i == n_batches - 1 else (i+1) * max_num_images
        ai_scores, _ = ai_score_fn(images=images[start_idx:end_idx], prompts="a cute cat", **ai_score_fn_args)
        ai_rewards += ai_scores.cpu().tolist()
        human_scores, _ = human_score_fn(images=images[start_idx:end_idx], prompts="a cute cat", **human_score_fn_args)
        human_rewards += human_scores

    return ai_rewards, human_rewards

def _normalize(input, input_range, normalize_range=(1,10)):
    input_range = np.array(input_range)
    normalize_range = np.array(normalize_range)
    scale = (normalize_range[1] - normalize_range[0]) / (input_range[1] - input_range[0])
    normalized = ((input - input_range.min()) * scale) + normalize_range[0]
    return normalized

def _normalize_chunkwise(inputs, chunk_size, normalize_range=(1,10)):
    n_inputs = inputs.shape[0]
    n_chunks = math.ceil(n_inputs / chunk_size)
    original_inputs = inputs.copy()
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = n_inputs if i == n_chunks - 1 else start_idx + chunk_size
        # min and max values up to this point
        running_min = original_inputs[:end_idx].min()
        running_max = original_inputs[:end_idx].max()
        # normalize this chunk
        inputs[start_idx:end_idx] = _normalize(input=original_inputs[start_idx:end_idx], input_range=(running_min, running_max), normalize_range=normalize_range)

    return inputs

def run_ai_feedback_correction_test(
        images, 
        featurizer_fn,
        featurizer_type,
        dataframe, 
        save_dir="ai_feedback_correction_outputs",
        experiment_name=None,
        chunk_size=20,
    ):
    reward_processor = RewardProcessor()
    n_samples = images.shape[0]
    n_chunks = math.ceil(n_samples / chunk_size)
    print("computing first batch of features")
    human_dataset_features = featurize(images=images[:chunk_size], featurizer_fn=featurizer_fn, featurizer_type=featurizer_type)
    # human_dataset_features = featurizer_fn(images[:chunk_size].float().to("cuda")).cpu()
    print("first batch of features added to human dataset")

    # values to record for plotting
    corrected_ai_feedback_errors = []
    corrected_ai_feedback_percent_errors = []
    ai_feedback_errors = []
    ai_feedback_percent_errors = []
    
    avg_min_distances = []

    human_rewards = []
    ai_rewards = []

    # normalized to 1-10 using min and max so far
    normalized_human_rewards = []
    normalized_ai_rewards = []
    normalized_corrected_ai_feedback = []

    # errors between normalized scores
    normalized_ai_feedback_errors = []
    normalized_ai_feedback_percent_errors = []
    normalized_corrected_ai_feedback_errors = []
    normalized_corrected_ai_feedback_percent_errors = []

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
        latents = featurizer_fn(images[start_idx:end_idx].float().to("cuda")).cpu()

        # compute similarity and AI feedback correction
        min_distances, most_similar_indices = reward_processor._get_most_similar_l2(features1=latents, features2=human_dataset_features)
        avg_min_distances.append(min_distances.mean())
        # dists = _compute_l2_distance(features1=latents, features2=human_dataset_features)

        # AI feedback 
        ai_rewards_for_this_batch = np.array(dataframe["ai_rewards"][start_idx:end_idx])
        ai_rewards.append(ai_rewards_for_this_batch.mean())
        normalized_ai_rewards_this_batch = np.array(dataframe["normalized_ai_rewards"][start_idx:end_idx])
        normalized_ai_rewards.append(normalized_ai_rewards_this_batch.mean())

        # Human feedback
        human_rewards_for_this_batch = np.array(dataframe["human_rewards"][start_idx:end_idx])
        human_rewards.append(human_rewards_for_this_batch.mean())
        normalized_human_rewards_this_batch = np.array(dataframe["normalized_human_rewards"][start_idx:end_idx])
        normalized_human_rewards.append(normalized_human_rewards_this_batch.mean())
        
        # Errors (raw)
        human_rewards_range = max(dataframe["human_rewards"][:end_idx]) - min(dataframe["human_rewards"][:end_idx]) # range of human rewards so far
        errors_for_this_batch = np.array(dataframe["errors"])[most_similar_indices]
        error_raw_ai_and_human = human_rewards_for_this_batch - ai_rewards_for_this_batch
        ai_feedback_errors.append(error_raw_ai_and_human.mean())
        ai_feedback_percent_errors.append((error_raw_ai_and_human / human_rewards_range).mean().tolist())
        
        ai_rewards_corrected = ai_rewards_for_this_batch + errors_for_this_batch
        error_corrected_ai_and_human = human_rewards_for_this_batch - ai_rewards_corrected
        corrected_ai_feedback_errors.append(error_corrected_ai_and_human.mean())
        percent_error = (error_corrected_ai_and_human / human_rewards_range).mean()
        corrected_ai_feedback_percent_errors.append(percent_error.tolist())

        # Errors (normalized)
        normalized_human_rewards_range = max(dataframe["normalized_human_rewards"][:end_idx]) - min(dataframe["normalized_human_rewards"][:end_idx]) # range of human rewards so far
        normalized_errors_for_this_batch = np.array(dataframe["normalized_errors"])[most_similar_indices]
        normalized_error_raw_ai_and_human = normalized_human_rewards_this_batch - normalized_ai_rewards_this_batch
        normalized_ai_feedback_errors.append(normalized_error_raw_ai_and_human.mean())
        normalized_ai_feedback_percent_errors.append((normalized_error_raw_ai_and_human / normalized_human_rewards_range).mean())

        normalized_ai_rewards_corrected = normalized_ai_rewards_this_batch + normalized_errors_for_this_batch
        normalized_corrected_ai_feedback.append(normalized_ai_rewards_corrected.mean())
        normalized_error_corrected_ai_and_human = normalized_human_rewards_this_batch - normalized_ai_rewards_corrected
        normalized_corrected_ai_feedback_errors.append(normalized_error_corrected_ai_and_human.mean())
        normalized_percent_error = (normalized_error_corrected_ai_and_human / normalized_human_rewards_range).mean()
        normalized_corrected_ai_feedback_percent_errors.append(normalized_percent_error.tolist())

        # Additional logging for plotting
        min_ai_rewards_this_batch.append(min(ai_rewards_for_this_batch))
        min_ai_rewards_overall.append(min(dataframe["ai_rewards"][:end_idx]))
        max_ai_rewards_this_batch.append(max(ai_rewards_for_this_batch))
        max_ai_rewards_overall.append(max(dataframe["ai_rewards"][:end_idx]))
        min_human_rewards_this_batch.append(min(human_rewards_for_this_batch))
        min_human_rewards_overall.append(min(dataframe["human_rewards"][:end_idx]))
        max_human_rewards_this_batch.append(max(human_rewards_for_this_batch))
        max_human_rewards_overall.append(max(dataframe["human_rewards"][:end_idx]))

        # add this current batch to the human dataset
        human_dataset_features = torch.cat([human_dataset_features, latents])
        n_human_feedback.append(start_idx)

    # Plot results
    experiment_name = experiment_name if experiment_name is not None else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Raw AI and human rewards
    print("saving human rewards and AI rewards (raw)")
    plt.plot(n_human_feedback, human_rewards, label="Human")
    plt.plot(n_human_feedback, ai_rewards, label="AI")
    plt.title("Human and AI Rewards (raw)")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "raw_ai_and_human_rewards.jpg"))
    plt.clf()

    # Normalized AI and human rewards
    print("saving human rewards and AI rewards (normalized)")
    plt.plot(n_human_feedback, normalized_human_rewards, label="Human")
    plt.plot(n_human_feedback, normalized_ai_rewards, label="AI")
    plt.title("Human and AI Rewards (normalized)")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Score")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "normalized_ai_and_human_rewards.jpg"))
    plt.clf()

    # Normalized Corrected AI and human rewards
    print("saving human rewards and AI rewards (normalized, corrected)")
    plt.plot(n_human_feedback, normalized_human_rewards, label="Human")
    plt.plot(n_human_feedback, normalized_corrected_ai_feedback, label="AI (corrected)")
    plt.title("Human and AI Rewards (normalized, corrected)")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Score")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "normalized_corrected_ai_and_human_rewards.jpg"))
    plt.clf()

    # Error between original AI feedback and human feedback
    print("saving AI-human error (raw)")
    plt.plot(n_human_feedback, ai_feedback_errors)
    plt.title("Error between Raw AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Error")
    plt.savefig(os.path.join(save_dir, "raw_ai_human_error.jpg"))
    plt.clf()

    # Normalized error between original AI feedback and human feedback
    print("saving normalized AI-human error (raw)")
    plt.plot(n_human_feedback, normalized_ai_feedback_errors)
    plt.title("Normalized error between Raw AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Error")
    plt.savefig(os.path.join(save_dir, "normalized_ai_human_error.jpg"))
    plt.clf()

    # Percent Error between corrected AI feedback and human feedback
    print("saving AI-human percent error")
    plt.plot(n_human_feedback, ai_feedback_percent_errors)
    plt.title("Percent Error between Raw AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Error")
    plt.savefig(os.path.join(save_dir, "raw_ai_human_percent_error.jpg"))
    plt.clf()

    # Normalized percent Error between corrected AI feedback and human feedback
    print("saving normalized AI-human percent error")
    plt.plot(n_human_feedback, normalized_ai_feedback_percent_errors)
    plt.title("Normalized percent Error between Raw AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Error")
    plt.savefig(os.path.join(save_dir, "normalized_ai_human_percent_error.jpg"))
    plt.clf()

    # Error between corrected AI feedback and human feedback
    print("saving AI-human error")
    plt.plot(n_human_feedback, corrected_ai_feedback_errors)
    plt.title("Error between Corrected AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Error")
    plt.savefig(os.path.join(save_dir, "raw_corrected_ai_human_error.jpg"))
    plt.clf()

    # Normalized error between corrected AI feedback and human feedback
    print("saving normalized AI(corrected)-human error")
    plt.plot(n_human_feedback, normalized_corrected_ai_feedback_errors)
    plt.title("Nromalized error between Corrected AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Error")
    plt.savefig(os.path.join(save_dir, "normalized_corrected_ai_human_error.jpg"))
    plt.clf()

    # Percent Error between corrected AI feedback and human feedback
    print("saving AI(corrected)-human percent error")
    plt.plot(n_human_feedback, corrected_ai_feedback_percent_errors)
    plt.title("Percent Error between Corrected AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Score Error")
    plt.savefig(os.path.join(save_dir, "raw_corrected_ai_human_percent_error.jpg"))
    plt.clf()

    # Percent Error between normalized and corrected AI feedback and human feedback
    print("saving AI(corrected)-human percent error (normalized)")
    plt.plot(n_human_feedback, normalized_corrected_ai_feedback_percent_errors)
    plt.title("Normalized percent Error between Corrected AI Feedback and Human Feedback")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Normalized Error")
    plt.savefig(os.path.join(save_dir, "normalized_corrected_ai_human_percent_error.jpg"))
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
    plt.fill_between(n_human_feedback, min_ai_rewards_this_batch, max_ai_rewards_this_batch, alpha=0.2, label="AI (this batch)")
    plt.fill_between(n_human_feedback, min_ai_rewards_overall, max_ai_rewards_overall, alpha=0.2, label="AI (overall)")
    plt.fill_between(n_human_feedback, min_human_rewards_this_batch, max_human_rewards_this_batch, alpha=0.2, label="Human (this batch)")
    plt.fill_between(n_human_feedback, min_human_rewards_overall, max_human_rewards_overall, alpha=0.2, label="Human (overall)")
    plt.title("AI and Human Feedback Range")
    plt.xlabel("N Human Feedback")
    plt.ylabel("Feedback Range")
    plt.legend(["AI (this batch)", "AI (overall)", "Human (this batch)", "Human (overall)"])
    plt.savefig(os.path.join(save_dir, "feedback_range.jpg"))
    plt.clf()

    print("Done saving figures")

    print("(mean, stdev) for errors")
    breakpoint()
    print(f"Normalized Human-AI(raw): ({np.array(normalized_ai_feedback_errors).mean()}, {np.std(np.array(normalized_ai_feedback_errors))})")
    print(f"Normalized Human-AI(corrected): ({np.array(normalized_corrected_ai_feedback_errors.mean())}, {np.std(np.array(normalized_corrected_ai_feedback_errors))})")


def main(args):
    # if dataframe file is provided, use this
    if args.dataframe is not None:
        print("Datafile was provided. Extracting data from pickle file...")
        df = pd.read_pickle(args.dataframe)
        img_paths, imgs = extract_images(image_paths=df["img_paths"].tolist())

    # if datafile is not provided, create one
    else:
        # initialize dataframe to save all info
        df = pd.DataFrame(
            columns=[
                "img_paths", 
                "human_rewards",
                "ai_rewards",
                "errors", # R_H - R_AI
                "normalized_human_rewards",
                "normalized_ai_rewards"
                "normalized_errors",
            ],
        )

        # initialize according to human and AI agents
        ai_evaluator, ai_evaluator_args = initialize_model(model_type=args.ai, device=args.device)
        human_evaluator, human_evaluator_args = initialize_model(model_type=args.human, device=args.device)

        # read images from input directory
        img_paths, imgs = extract_images(input_dir=args.input_dir, max_images_per_epoch=args.n_imgs_per_epoch)

        # get AI and human rewards
        ai_rewards, human_rewards = get_reward_labels(
            images=imgs,
            ai_score_fn=ai_evaluator,
            ai_score_fn_args=ai_evaluator_args,
            human_score_fn=human_evaluator,
            human_score_fn_args=human_evaluator_args,
        )
        normalize_range = (1, 10)
        normalized_ai_rewards = _normalize_chunkwise(inputs=np.array(ai_rewards), chunk_size=args.chunk_size, normalize_range=normalize_range)
        normalized_human_rewards = _normalize_chunkwise(inputs=np.array(human_rewards), chunk_size=args.chunk_size, normalize_range=normalize_range)

        df["img_paths"] = img_paths
        df["human_rewards"] = human_rewards
        df["ai_rewards"] = ai_rewards
        df["errors"] = np.array(human_rewards) - np.array(ai_rewards) 
        df["normalized_human_rewards"] = normalized_human_rewards
        df["normalized_ai_rewards"] = normalized_ai_rewards
        df["normalized_errors"] = normalized_human_rewards - normalized_ai_rewards
        print("human and ai dataframe is ready")

        df.to_pickle("/home/ayanoh/all_aesthetics.pkl")
        print("Saved dataframe as pickle file")

    # run ai feedback correction test
    # initialize featurizer model
    featurizer_fn = initialize_featurizer(featurizer_type=args.featurizer)
    run_ai_feedback_correction_test(
        images=imgs, 
        featurizer_fn=featurizer_fn, 
        featurizer_type=args.featurizer,
        chunk_size=args.chunk_size, 
        save_dir=args.save_dir, 
        experiment_name=args.name,
        dataframe=df,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--ai", type=str, default="laion")
    parser.add_argument("--human", type=str, default="pickscore")
    parser.add_argument("--featurizer", type=str, default="sd")
    parser.add_argument("--save-dir", type=str, default="/home/ayanoh/ai_correction_experiments")
    parser.add_argument("--n-imgs-per-epoch", type=int)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str)
    parser.add_argument("--dataframe", type=str, help="if using a saved pickle file containing all human and ai feedback data, provide path. It saves time.")

    args = parser.parse_args()
    main(args)