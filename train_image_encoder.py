
import os
import math
import time

import random
import numpy as np
import torch 
from torch import nn
import torchvision
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

import argparse
import wandb
import datetime
import pandas as pd
from PIL import Image

from rl4dgm.models.my_models import LinearModel
from rl4dgm.models.mydatasets import HumanRewardDataset, TripletDataset

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
                # images.append(Image.open(filepath))
                n_saved_images += 1
    else:
        images = []
        for filepath in image_paths:
            n_saved_images = 0
            if max_images_per_epoch is not None and n_saved_images >= max_images_per_epoch:
                break
            images.append(torchvision.io.read_image(filepath))
            # images.append(Image.open(filepath))
            n_saved_images += 1
    
    return images
    # return torch.stack(images)

def get_datasets(features, scores, n_samples_per_epoch, train_ratio, batch_size, device):
    """
    Create train and test datasets
    Args:
        features (torch.Tensor) : images or features
        scores (toorch.Tensor) : scores for images
        n_samples_per_epoch (int) : number of samples per epoch (so images from each epoch is separated into train and test)
        train_ratio (float) : ratio of data to use as train data
        batch_size (int) : batch size for dataloaders
    """
    assert features.shape[0] % n_samples_per_epoch == 0, "number of features is not a multiple of n_samples_per_epoch"
    n_epochs = int(features.shape[0] / n_samples_per_epoch)
    n_train_per_epoch = int(n_samples_per_epoch * train_ratio)
    indices = np.arange(n_samples_per_epoch)
    random.shuffle(indices)
    train_indices_per_epoch = indices[:n_train_per_epoch]
    test_indices_per_epoch = indices[n_train_per_epoch:]
    train_indices = []
    test_indices = []
    for i in range(n_epochs):
        train_indices += (train_indices_per_epoch + i * n_samples_per_epoch).tolist() 
        test_indices += (test_indices_per_epoch + i * n_samples_per_epoch).tolist() 

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_scores = torch.tensor(scores[train_indices].values)
    test_scores = torch.tensor(scores[test_indices].values)

    print("Split into train and test")
    print("Initializing train set")
    trainset = TripletDataset(
        features=train_features,
        scores=train_scores,
        device=device,
        is_train=True,
    )

    print("Initializing test set")
    testset = TripletDataset(
        features=test_features,
        scores=test_scores,
        device=device,
        is_train=False,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader, train_features, test_features, train_scores, test_scores

def train(model, trainloader, testloader, n_epochs=100, lr=0.001):
    
    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(p=2, margin=1.0)

    n_steps = 0
    start_time = time.time()
    for epoch in range(n_epochs):
        running_losses = []
        print("epoch", epoch)
        ###############################################################################
        # Train
        ###############################################################################
        for step, (anchor_features, anchor_scores, positive_features, negative_features) in enumerate(trainloader):
            optimizer.zero_grad()
            anchor_out = model(anchor_features)
            positive_out = model(positive_features)
            negative_out = model(negative_features)

            loss = criterion(x=anchor_out, positive=positive_out, negative=negative_out)
            # print("loss", loss.item())
            loss.backward()
            optimizer.step()
            running_losses.append(loss.item())
            n_steps += 1

            wandb.log({
                "epoch" : epoch,
                "step" : n_steps,
                "loss" : loss.item(),
                "lr" : lr,
                "clock_time" : time.time() - start_time,
            })

        ###############################################################################
        # Test
        ###############################################################################
        with torch.no_grad():
            # trainset
            anchor_positive = []
            anchor_negative = []
            for step, (anchor_features, anchor_scores, positive_features, negative_features) in enumerate(trainloader):
                # get anchor-positive and anchor-negative distances
                anchor_positive_dist = torch.linalg.norm(anchor_out - positive_out)
                anchor_positive.append(anchor_positive_dist.mean().item())
                anchor_negative_dist = torch.linalg.norm(anchor_out - negative_out)
                anchor_negative.append(anchor_negative_dist.mean().item())
                
            wandb.log({
                "train_anchor_positive_dist" : np.array(anchor_positive).mean(),
                "train_anchor_negative_dist" : np.array(anchor_negative).mean(),
            })

            # testset
            anchor_positive = []
            anchor_negative = []
            for step, (anchor_features, anchor_scores, positive_features, negative_features) in enumerate(testloader):
                # get anchor-positive and anchor-negative distances
                anchor_positive_dist = torch.linalg.norm(anchor_out - positive_out)
                anchor_positive.append(anchor_positive_dist.mean().item())
                anchor_negative_dist = torch.linalg.norm(anchor_out - negative_out)
                anchor_negative.append(anchor_negative_dist.mean().item())
                
            wandb.log({
                "test_anchor_positive_dist" : np.array(anchor_positive).mean(),
                "test_anchor_negative_dist" : np.array(anchor_negative).mean(),
            })

            
def main(args):

    # set seed
    torch.manual_seed(0)

    # extract images
    images = extract_images(input_dir=args.img_dir)
    images = torch.stack(images)
    print("extracted images")

    # get labels
    df = pd.read_pickle(args.datafile)
    human_rewards = df["human_rewards"]
    ai_rewards = df["ai_rewards"]

    # setup datasets
    trainloader, testloader, train_features, test_features, train_human_rewards, test_human_rewards = get_datasets(
        features=images,
        scores=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    # setup model
    # flatten the features if using linear model
    features = torch.flatten(images, start_dim=1)
    model = LinearModel(
        input_dim=features.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        device=args.device,
    )
    model = model.to(args.device)
    print("Initialized model")

    # setup wandb for logging
    # ai_feedback_dim = str(args.ai_feedback_dim) if args.use_ai_feedback else "none"
    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=args.experiment+f"AIencoder"+datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        project="encoder training",
        entity="misoshiruseijin",
        config={
            "input_dim" : features.shape[1],
            "hidden_dims" : args.hidden_dims,
            "n_hidden_layers" : args.n_hidden_layers,
            "output_dim" : args.output_dim,
            "lr" : args.lr,
            "datafile" : args.datafile,
            "train_ratio" : args.train_ratio,
            "n_epochs" : args.n_epochs,
            "batch_size" : args.batch_size,
        }
    )

    # train
    train(
        model=model,
        trainloader=trainloader, 
        testloader=testloader, 
        train_features=train_features.to(args.device), 
        test_features=test_features.to(args.device), 
        train_human_rewards=train_human_rewards.to(args.device),
        test_human_rewards=test_human_rewards.to(args.device),
        n_epochs=args.n_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/ayanoh/all_aesthetic_small")
    parser.add_argument("--datafile", type=str, default="/home/ayanoh/all_aesthetic.pkl")
    parser.add_argument("--save-dir", type=str, default="/home/ayanoh/reward_model_training_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[22000]*6)
    parser.add_argument("--n-hidden-layers", type=int, default=5)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--experiment", type=str, default="")

    args = parser.parse_args()
    main(args)