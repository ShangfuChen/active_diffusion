
"""
OUTDATED AND BUGGY DON'T USE
"""


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

from rl4dgm.models.reward_predictor_model import RewardPredictorModel, RewardPredictorModel2, RewardPredictorModelCNN
from rl4dgm.models.mydatasets import HumanRewardDataset


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

def get_datasets(features, human_rewards, ai_rewards, n_samples_per_epoch, train_ratio, batch_size, device):
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
    train_indices = []
    test_indices = []
    for i in range(n_epochs):
        train_indices += (train_indices_per_epoch + i * n_samples_per_epoch).tolist() 
        test_indices += (test_indices_per_epoch + i * n_samples_per_epoch).tolist() 

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_human_rewards = torch.tensor(human_rewards[train_indices].values)
    test_human_rewards = torch.tensor(human_rewards[test_indices].values)
    train_ai_rewards = torch.tensor(ai_rewards[train_indices].values)
    test_ai_rewards = torch.tensor(ai_rewards[test_indices].values)

    print("Split into train and test")
    print("Initializing train set")
    trainset = HumanRewardDataset(
        features=train_features, 
        human_rewards=train_human_rewards, 
        ai_rewards=torch.tensor(ai_rewards[train_indices].values),
        device=device,
    )
    print("Initializing test set")
    testset = HumanRewardDataset(
        features=test_features,
        human_rewards=test_human_rewards, 
        ai_rewards=torch.tensor(ai_rewards[test_indices].values),
        device=device,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader, train_features, test_features, train_human_rewards, test_human_rewards, train_ai_rewards, test_ai_rewards

def train(model_ai,
          model_human,
          trainloader,
          testloader,
          train_features,
          test_features,
          train_human_rewards,
          test_human_rewards,
          train_ai_rewards,
          test_ai_rewards,
          n_epochs=100,
          lr=0.001):
    
    # initialize optimizer and loss function
    optimizer_ai = torch.optim.Adam(model_ai.parameters(), lr=lr)
    optimizer_human = torch.optim.Adam(model_human.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    n_steps = 0
    start_time = time.time()
    for epoch in range(n_epochs):
        running_losses = []
        print("epoch", epoch)
        ###############################################################################
        # Train
        ###############################################################################
        for step, (features, human_rewards, ai_rewards) in enumerate(trainloader):
            optimizer_ai.zero_grad()
            optimizer_human.zero_grad()

            # TODO - currently only feeding feature in. should be using ai feedback as well
            features_ai = model_ai(features)
            features_human = model_human(features)
            predictions = (features_ai - features_human).mean(-1)
            # print("predictions", predictions)
            # print("ground truth", human_rewards)
            # loss = torch.sqrt(criterion(predictions, human_rewards))
            # loss = torch.sqrt(criterion(predictions, torch.abs(human_rewards - ai_rewards)))
            loss = criterion(predictions, torch.abs(human_rewards - ai_rewards))

            # print("loss", loss.item())
            loss.backward()
            optimizer_ai.step()
            optimizer_human.step()
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
            train_features_ai = model_ai(train_features)
            train_features_human = model_human(train_features)
            train_outputs = (train_features_ai - train_features_human).mean(-1).cpu()
            test_features_ai = model_ai(test_features)
            test_features_human = model_human(test_features)
            test_outputs = (test_features_ai - test_features_human).mean(-1).cpu()
            # train_outputs = torch.flatten(model_ai(train_features).cpu())
            # test_outputs = torch.flatten(model(test_features).cpu())
            # train_labels = train_human_rewards.cpu()
            # test_labels = test_human_rewards.cpu()
            train_labels = torch.abs(train_human_rewards - train_ai_rewards).cpu()
            test_labels = torch.abs(test_human_rewards - test_ai_rewards).cpu()

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
    
    # normalize
    human_rewards = (human_rewards - human_rewards.min())/(human_rewards.max()-human_rewards.min())
    ai_rewards = (ai_rewards - ai_rewards.min())/(ai_rewards.max()-ai_rewards.min())

    # set up feature extraction function
    # pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    # pretrained_revision = "main"
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     pretrained_model_path, 
    #     revision=pretrained_revision
    # ).to(args.device)

    # pipeline.vae.requires_grad_(False)
    # pipeline.text_encoder.requires_grad_(False)
    # pipeline.unet.requires_grad_(False)
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # featurizer_fn = pipeline.vae.encoder

    # get image features in increments
    # features = featurizer_fn(images.float().to(args.device)).cpu()
    assert os.path.exists(args.featurefile)
    # if not os.path.exists(args.featurefile):
    #     # extract imagesi
    #     print("Extracting images...")
    #     images = extract_images(input_dir=args.img_dir)
    #     max_n_imgs = 32
    #     n_batches = math.ceil(images.shape[0] / max_n_imgs)
    #     print("Extracting features...")
    #     features = []
    #     for i in range(n_batches):
    #         print(f"{i+1} / {n_batches}")
    #         start_idx = i * max_n_imgs
    #         end_idx = images.shape[0] if i == n_batches - 1 else start_idx + max_n_imgs
    #         features.append(featurizer_fn(images[start_idx:end_idx].float().to(args.device)).cpu())
    #     features = torch.cat(features)
    #     print("...done")
    #     torch.save(features, args.featurefile)
    # else:
    features = torch.load(args.featurefile)
    # flatten features and get shape (input_dim for reward prediction model)
    # features = torch.flatten(features, start_dim=1)
    feature_shape = features.shape
    print("Flattened features to shape", feature_shape)

    if args.use_ai_feedback:
        # expand ai feedback to same dimension as feature and concatenate
        expanded_ai_feedback = torch.ones(args.ai_feedback_dim).unsqueeze(0).expand(feature_shape[0], -1)
        features = torch.cat([features, expanded_ai_feedback], dim=1)
        feature_shape = features.shape
        print(f"Concatenated AI feedback (dim={args.ai_feedback_dim})")
        print("New feature shape is ", feature_shape)

    # setup datasets
    trainloader, testloader, train_features, test_features, train_human_rewards, \
        test_human_rewards, train_ai_rewards, test_ai_rewards = get_datasets(
        features=features,
        human_rewards=human_rewards,
        ai_rewards=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    # setup reward prediction model to train
    # model = RewardPredictorModel(
    #     input_dim=feature_shape[1], 
    #     hidden_dim=args.hidden_dim,
    #     n_hidden_layers=args.n_hidden_layers,
    #     device=args.device,
    # )

    # model = RewardPredictorModel(
    #     input_dim=feature_shape[1], 
    #     hidden_dims=args.hidden_dims,
    #     n_hidden_layers=args.n_hidden_layers,
    #     device=args.device,
    # )

    model_ai = RewardPredictorModelCNN(device=args.device,)
    model_human = RewardPredictorModelCNN(device=args.device,)

    # model_ai = RewardPredictorModel2(input_dim=feature_shape[1], device=args.device,)
    # model_human = RewardPredictorModel2(input_dim=feature_shape[1], device=args.device,)
    print("initialized model")

    # setup wandb for logging
    ai_feedback_dim = str(args.ai_feedback_dim) if args.use_ai_feedback else "none"
    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=args.experiment+f"ai_dim{ai_feedback_dim}_hidden_dims{args.hidden_dims}_"+datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        project="reward_predictor_training",
        entity="sfchen",
        config={
            "input_dim" : feature_shape[1],
            "hidden_dims" : args.hidden_dims,
            "n_hidden_layers" : args.n_hidden_layers,
            "lr" : args.lr,
            "datafile" : args.datafile,
            "train_ratio" : args.train_ratio,
            "n_epochs" : args.n_epochs,
            "batch_size" : args.batch_size,
            "use_ai_feedback" : args.use_ai_feedback,
            "ai_feedback_dim" : args.ai_feedback_dim,
            "feature_shape" : feature_shape,
        }
    )

    # train
    train(
        model_ai=model_ai,
        model_human=model_human,
        trainloader=trainloader, 
        testloader=testloader, 
        train_features=train_features.to(args.device), 
        test_features=test_features.to(args.device), 
        train_human_rewards=train_human_rewards.to(args.device),
        test_human_rewards=test_human_rewards.to(args.device),
        train_ai_rewards=train_ai_rewards.to(args.device),
        test_ai_rewards=test_ai_rewards.to(args.device),
        n_epochs=args.n_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/shangfu/all_aesthetic")
    parser.add_argument("--datafile", type=str, default="/home/shangfu/all_aesthetic.pkl")
    parser.add_argument("--featurefile", type=str, default="/home/shangfu/all_aesthetic_feature.pt")
    parser.add_argument("--save-dir", type=str, default="/home/shangfu/reward_model_training_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[8192, 2048, 512])
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--use-ai-feedback", action="store_true")
    parser.add_argument("--ai-feedback-dim", type=int, default=2048)
    parser.add_argument("--experiment", type=str, default="")

    args = parser.parse_args()
    main(args)