
import os
import math
import time

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

from rl4dgm.models.reward_predictor_model import RewardPredictorModel, RewardPredictorModel2
from active_diffusion.rl4dgm.models.mydatasets import HumanRewardDatasetNoImage


# def extract_images(input_dir=None, image_paths=None, max_images_per_epoch=None,):
#     assert sum([input_dir is not None, image_paths is not None]) == 1, "Either input_dir or image_paths should be provided (but not both)"
#     if input_dir is not None:
#         image_paths = [] # image paths
#         images = [] # image tensors
#         for epoch in os.listdir(input_dir):
#             # print("storing images for epoch", epoch)
#             n_saved_images = 0
#             for img in os.listdir(os.path.join(input_dir, epoch)):
#                 if max_images_per_epoch is not None and n_saved_images >= max_images_per_epoch:
#                     break
#                 filepath = os.path.join(input_dir, epoch, img)
#                 image_paths.append(filepath)
#                 images.append(torchvision.io.read_image(filepath))
#                 n_saved_images += 1
#     else:
#         images = []
#         for filepath in image_paths:
#             n_saved_images = 0
#             if max_images_per_epoch is not None and n_saved_images >= max_images_per_epoch:
#                 break
#             images.append(torchvision.io.read_image(filepath))
#             n_saved_images += 1
    
#     return torch.stack(images)

def get_datasets(human_rewards, ai_rewards, n_samples_per_epoch, train_ratio, batch_size, device):
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
    assert human_rewards.shape[0] % n_samples_per_epoch == 0, "number of human_rewards is not a multiple of n_samples_per_epoch"
    n_epochs = int(human_rewards.shape[0] / n_samples_per_epoch)
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

    train_human_rewards = torch.tensor(human_rewards[train_indices].values)
    test_human_rewards = torch.tensor(human_rewards[test_indices].values)
    train_ai_rewards = torch.tensor(ai_rewards[train_indices].values)
    test_ai_rewards = torch.tensor(ai_rewards[test_indices].values)

    print("Split into train and test")
    print("Initializing train set")
    trainset = HumanRewardDatasetNoImage(
        human_rewards=train_human_rewards, 
        ai_rewards=train_ai_rewards,
        device=device,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader, train_human_rewards, test_human_rewards, train_ai_rewards, test_ai_rewards

def train(
        model, trainloader, 
        train_human_rewards, test_human_rewards, 
        train_ai_rewards, test_ai_rewards, 
        n_epochs=100, lr=0.001
    ):
    
    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_steps = 0
    start_time = time.time()
    for epoch in range(n_epochs):
        running_losses = []
        print("epoch", epoch)
        ###############################################################################
        # Train
        ###############################################################################
        for step, (human_rewards, ai_rewards) in enumerate(trainloader):
            ai_rewards = ai_rewards[:,None].float()
            optimizer.zero_grad()
            predictions = torch.flatten(model(ai_rewards))
            loss = criterion(predictions, human_rewards)
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
            train_outputs = torch.flatten(model(train_ai_rewards[:,None].float()).cpu())
            test_outputs = torch.flatten(model(test_ai_rewards[:,None].float()).cpu())
            train_labels = train_human_rewards.cpu()
            test_labels = test_human_rewards.cpu()

            train_errors = np.abs(train_outputs - train_labels)
            test_errors = np.abs(test_outputs - test_labels)

            train_percent_error = train_errors / (train_errors.max() - train_errors.min())
            test_percent_error = test_errors / (test_errors.max() - test_errors.min())

            wandb.log({
                "train_mean_error" : train_errors.mean(),
                "train_mean_percent_error" : train_percent_error.mean(),
                "train_samples_with_under_10%_error" : (train_percent_error < 0.1).sum() / train_outputs.shape[0],
                "train_samples_with_under_20%_error" : (train_percent_error < 0.2).sum() / train_outputs.shape[0],
                "train_samples_with_under_30%_error" : (train_percent_error < 0.3).sum() / train_outputs.shape[0],
                "train_samples_with_under_40%_error" : (train_percent_error < 0.4).sum() / train_outputs.shape[0],
                "train_samples_with_under_50%_error" : (train_percent_error < 0.5).sum() / train_outputs.shape[0],
                "test_mean_error" : test_errors.mean(),
                "test_mean_percent_error" : test_percent_error.mean(),
                "test_samples_with_under_10%_error" : (test_percent_error < 0.1).sum() / test_outputs.shape[0],
                "test_samples_with_under_20%_error" : (test_percent_error < 0.2).sum() / test_outputs.shape[0],
                "test_samples_with_under_30%_error" : (test_percent_error < 0.3).sum() / test_outputs.shape[0],
                "test_samples_with_under_40%_error" : (test_percent_error < 0.4).sum() / test_outputs.shape[0],
                "test_samples_with_under_50%_error" : (test_percent_error < 0.5).sum() / test_outputs.shape[0],
            })
            
def main(args):

    # set seed
    torch.manual_seed(0)

    # get labels
    df = pd.read_pickle(args.datafile)
    human_rewards = df["human_rewards"]
    ai_rewards = df["ai_rewards"]

    # setup datasets
    trainloader, train_human_rewards, test_human_rewards, train_ai_rewards, test_ai_rewards = get_datasets(
        human_rewards=human_rewards,
        ai_rewards=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    # setup reward prediction model
    model = RewardPredictorModel2(
        input_dim=1, 
        hidden_dims=args.hidden_dims,
        n_hidden_layers=args.n_hidden_layers,
        device=args.device,
    )
    print("initialized model")

    # setup wandb for logging
    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=args.experiment+f"hidden_dim{args.hidden_dims}_"+datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        project="reward_predictor_training",
        entity="misoshiruseijin",
        config={
            "input_dim" : 1,
            "hidden_dim" : args.hidden_dims,
            "n_hidden_layers" : args.n_hidden_layers,
            "lr" : args.lr,
            "datafile" : args.datafile,
            "train_ratio" : args.train_ratio,
            "n_epochs" : args.n_epochs,
            "batch_size" : args.batch_size,
            "feature_shape" : 1,
        }
    )

    # train
    train(
        model=model,
        trainloader=trainloader, 
        train_human_rewards=train_human_rewards.to(args.device),
        test_human_rewards=test_human_rewards.to(args.device),
        train_ai_rewards=train_ai_rewards.to(args.device),
        test_ai_rewards=test_ai_rewards.to(args.device),
        n_epochs=args.n_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--img-dir", type=str, default="/home/ayanoh/all_aesthetic")
    parser.add_argument("--datafile", type=str, default="/home/ayanoh/all_aesthetic.pkl")
    parser.add_argument("--save-dir", type=str, default="/home/ayanoh/reward_model_training_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[64, 64, 64])
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--experiment", type=str, default="")

    args = parser.parse_args()
    main(args)