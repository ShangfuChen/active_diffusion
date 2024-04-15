
import os

import random
import numpy as np
import torch 
from torch import nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, Dataset

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel

import argparse
import wandb
import datetime
import pandas as pd

from rl4dgm.models.reward_predictor_model import RewardPredictorModel
from rl4dgm.models.human_reward_dataset import HumanRewardDataset


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
    
    return torch.stack(images)

def get_datasets(features, human_rewards, ai_rewards, n_samples_per_epoch, train_ratio, batch_size):
    """
    Create train and test datasets
    Args:
        features (torch.Tensor) : image features
        human_rewards : human reward labels corresponding to the features
        ai_rewards : ai reward labels corresponding to the features
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
    train_indices = [train_indices_per_epoch + i * n_samples_per_epoch for i in range(n_epochs)]
    test_indices = [test_indices_per_epoch + i * n_samples_per_epoch for i in range(n_epochs)]
    trainset = HumanRewardDataset(features=features[train_indices], human_rewards=human_rewards[train_indices], ai_rewards=ai_rewards[train_indices])
    testset = HumanRewardDataset(features=features[test_indices], human_rewards=human_rewards[test_indices], ai_rewards=ai_rewards[test_indices])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

def train(model, trainloader, testloader, n_epochs=100):
    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        running_losses = []
        print("epoch", epoch)
        ###############################################################################
        # Train
        ###############################################################################
        for step, (features, ai_rewards, human_rewards) in enumerate(trainloader):
            print("step", step)

            optimizer.zero_grad()
            # TODO - currently only feeding feature in. should be using ai feedback as well
            predictions = model(features)
            loss = criterion(predictions, human_rewards)
            loss.backward()
            optimizer.step()
            running_losses.append(loss.item())

        ###############################################################################
        # Test
        ###############################################################################
        with torch.no_grad():
            train_outputs = model(trainloader.features)
            test_outputs = model(testloader.features)
            # TODO - compute distance between outputs and human rewards for each (train and test) datasets
            # TODO - compute train and test accuracies

def main(args):

    # setup wandb logging
    wandb.init(
        name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        project="reward_predictor_training",
        entity="misoshiruseijin",
    )

    # extract images
    images = extract_images(input_dir=args.img_dir)

    # get labels
    df = pd.read_pickle(args.datafile)
    human_rewards = df["human_rewards"]
    ai_rewards = df["ai_rewards"]

    # set up feature extraction function
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

    # get image features
    features = featurizer_fn(images.float().to(args.device)).cpu()
    # TODO - flatten features and get shape (input_dim for reward prediction model)

    # setup datasets
    trainloader, testloader = get_datasets(
        features=features,
        human_rewards=human_rewards,
        ai_rewards=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
    )

    # setup reward prediction model to train
    model = RewardPredictorModel(
        input_dim=0, # TODO
        hidden_dim=1024,
        n_hidden_layers=5,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--datafile", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, deafult=8)

    args = parser.parse_args()
    main(args)