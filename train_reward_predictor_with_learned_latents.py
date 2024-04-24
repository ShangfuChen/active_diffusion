
import os
import math
import time

import json
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

from rl4dgm.models.my_models import LinearModel
from rl4dgm.models.mydatasets import HumanRewardDataset, FeatureLabelDataset


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

def get_datasets(features, rewards, n_samples_per_epoch, train_ratio, batch_size, device):
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
    train_rewards = torch.tensor(rewards[train_indices].values)
    test_rewards = torch.tensor(rewards[test_indices].values)

    print("Split into train and test")
    print("Initializing train set")
    trainset = FeatureLabelDataset(
        features=train_features,
        labels=train_rewards,
        device=device,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, train_features, test_features, train_rewards, test_rewards

def train(model, trainloader, train_features, test_features, train_rewards, test_rewards, model_save_dir, save_every, n_epochs=100, lr=0.001):
    
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
        for step, (features, rewards) in enumerate(trainloader):
            optimizer.zero_grad()
            # TODO - currently only feeding feature in. should be using ai feedback as well
            predictions = model(features)
            # print("predictions", predictions)
            # print("ground truth", human_rewards)
            loss = criterion(predictions, rewards[:,None])
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
            train_outputs = torch.flatten(model(train_features).cpu())
            test_outputs = torch.flatten(model(test_features).cpu())
            train_labels = train_rewards.cpu()
            test_labels = test_rewards.cpu()

            train_errors = np.abs(train_outputs - train_labels)
            test_errors = np.abs(test_outputs - test_labels)

            train_percent_error = train_errors / (train_labels.max() - train_labels.min())
            test_percent_error = test_errors / (test_labels.max() - test_labels.min())

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

        if (epoch > 0) and (epoch % save_every) == 0:
            print("Saving model checkpoint...")
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch{epoch}.pt"))
            print("done")
    
    print("Saving model checkpoint...")
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch{epoch}.pt"))
    print("done")
            
def main(args):

    # set seed
    torch.manual_seed(0)

    # experiment time
    experiment_time = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    # make save directory
    save_dir = os.path.join(args.save_dir, f"{args.agent}_{args.experiment}", experiment_time)
    os.makedirs(save_dir, exist_ok=False)

    # get labels
    df = pd.read_pickle(args.datafile)
    # human_rewards = df["human_rewards"]
    # ai_rewards = df["ai_rewards"]
    rewards = df[f"{args.agent}_rewards"]

    # read features file
    features = torch.load(args.featurefile)
    features = torch.flatten(features, start_dim=1)

    # load AI encoder model
    with open(os.path.join(args.pretrained_encoder_conf)) as f:
        encoder_conf = json.load(f)
    encoder_model = LinearModel(
        input_dim=encoder_conf["input_dim"],
        hidden_dims=encoder_conf["hidden_dims"],
        output_dim=encoder_conf["output_dim"],
        device=args.device,
    )
    encoder_model.load_state_dict(torch.load(args.pretrained_encoder_state_dict))
    encoder_model.to(args.device)
    print(encoder_model)
    print("loaded AI encoder model")
    
    # encode features
    with torch.no_grad():
        encoded_features = encoder_model(features.to(args.device)).cpu()
        encoded_feature_shape = encoded_features.shape
        print("encoded features. feature shape ", encoded_feature_shape)

    # setup datasets
    trainloader, train_features, test_features, train_rewards, test_rewards = get_datasets(
        features=encoded_features,
        # features=features,
        rewards=rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    # setup reward prediction model to train
    reward_prediction_model = LinearModel(
        input_dim=encoded_features.shape[1],
        # input_dim=features.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        device=args.device,
    )
    reward_prediction_model = reward_prediction_model.to(args.device)
    print("initialized model")

    # setup wandb for logging
    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=f"{args.agent}_"+args.experiment+f"hidden_dims{args.hidden_dims}_"+experiment_time,
        project="reward_predictor_training",
        entity="misoshiruseijin",
        config={
            "input_dim" : encoded_features.shape[1],
            # "input_dim" : features.shape[1],
            "hidden_dims" : args.hidden_dims,
            "n_hidden_layers" : args.n_hidden_layers,
            "lr" : args.lr,
            "datafile" : args.datafile,
            "train_ratio" : args.train_ratio,
            "n_epochs" : args.n_epochs,
            "batch_size" : args.batch_size,
        }
    )

    # train
    train(
        model=reward_prediction_model,
        trainloader=trainloader,
        train_features=train_features.to(args.device),
        test_features=test_features.to(args.device),
        train_rewards=train_rewards.to(args.device),
        test_rewards=test_rewards.to(args.device),
        n_epochs=args.n_epochs,
        lr=args.lr,
        model_save_dir=save_dir,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/hayano/all_aesthetic")
    parser.add_argument("--datafile", type=str, default="/home/hayano/all_aesthetic.pkl")
    parser.add_argument("--save-dir", type=str, default="/home/hayano/reward_model_training_results")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[2048]*6)
    parser.add_argument("--n-hidden-layers", type=int, default=5)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--experiment", type=str, default="reward_prediction")
    parser.add_argument("--featurefile", type=str, default="/home/hayano/all_aesthetic_feature.pt")
    parser.add_argument("--agent", type=str, help="which agent's rewards to use for encoder training - ai or human")
    parser.add_argument("--pretrained-encoder-state-dict", type=str, help="path to state dict pt file")
    parser.add_argument("--pretrained-encoder-conf", type=str, help="path to pretrained encoder config json")

    args = parser.parse_args()
    main(args)