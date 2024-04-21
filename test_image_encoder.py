
import os
import math
import time

import random
import json
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
from rl4dgm.models.mydatasets import HumanRewardDataset, TripletDataset, FeatureLabelDataset

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
    print("Initializing test set")
    testset = FeatureLabelDataset(
        features=test_features,
        labels=test_rewards, 
        device=device,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader, train_features, test_features, train_rewards, test_rewards

def train(model, trainloader, train_features, test_features, train_rewards, test_rewards, n_epochs=100, lr=0.001, model_save_dir="image_encoder", save_every=10):
    
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
        for step, (features, reward) in enumerate(trainloader):
            optimizer.zero_grad()
            predictions = model(features)
            print("predictions", predictions)
            print("ground truth", reward)
            loss = criterion(predictions, reward[:,None])
            print("loss", loss.item())
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

    # create directory to save model 
    save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=False)
    
    if not os.path.exists(args.featurefile):
        # extract images
        images = extract_images(input_dir=args.img_dir)
        images = torch.stack(images)
        print("extracted images")
        
        # set up feature extraction function
        pretrained_model_path = "runwayml/stable-diffusion-v1-5"
        pretrained_revision = "main"
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path, 
            revision=pretrained_revision
        ).to(args.device)

        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        featurizer_fn = pipeline.vae.encoder

        # get image features in increments
        # features = featurizer_fn(images.float().to(args.device)).cpu()
        max_n_imgs = 32
        n_batches = math.ceil(images.shape[0] / max_n_imgs)
        print("Extracting features...")
        features = []
        for i in range(n_batches):
            print(f"{i+1} / {n_batches}")
            start_idx = i * max_n_imgs
            end_idx = images.shape[0] if i == n_batches - 1 else start_idx + max_n_imgs
            features.append(featurizer_fn(images[start_idx:end_idx].float().to(args.device)).cpu())
        features = torch.cat(features)
        print("...done")
        torch.save(features, args.featurefile)

        # flatten features and get shape (input_dim for reward prediction model)
        # features = torch.flatten(features, start_dim=1)
    
    else:
        print("loaded features")
        features = torch.load(args.featurefile)
    
    features = torch.flatten(features, start_dim=1)
    feature_shape = features.shape
    print("Flattened features to shape", feature_shape)
    # breakpoint()

    # load AI encoder model
    with open(os.path.join(args.encoder_model_dir, "train_config.json")) as f:
        encoder_conf = json.load(f)
    ai_encoder = LinearModel(
        input_dim=encoder_conf["input_dim"],
        hidden_dims=encoder_conf["hidden_dims"],
        output_dim=encoder_conf["output_dim"],
        device=args.device,
    )
    ai_encoder.load_state_dict(torch.load(os.path.join(args.encoder_model_dir, "epoch99.pt")))
    ai_encoder.eval()
    ai_encoder.to(args.device)
    print(ai_encoder)
    print("loaded AI encoder model")
    
    # encode features
    ai_features = ai_encoder(features.to(args.device))
    ai_feature_shape = ai_features.shape
    print("encoded features. feature shape ", ai_feature_shape)


    # get labels
    df = pd.read_pickle(args.datafile)
    human_rewards = df["human_rewards"]
    ai_rewards = df["ai_rewards"]

    # setup datasets
    trainloader, testloader, train_features, test_features, train_rewards, test_rewards = get_datasets(
        features=ai_features,
        rewards=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    # setup model
    # flatten the features if using linear model
    model = LinearModel(
        input_dim=ai_features.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        device=args.device,
    )
    model = model.to(args.device)

    print("Initialized model")

    # setup wandb for logging
    # ai_feedback_dim = str(args.ai_feedback_dim) if args.use_ai_feedback else "none"
    train_config={
        "input_dim" : ai_features.shape[1],
        "hidden_dims" : args.hidden_dims,
        "n_hidden_layers" : args.n_hidden_layers,
        "output_dim" : args.output_dim,
        "lr" : args.lr,
        "datafile" : args.datafile,
        "train_ratio" : args.train_ratio,
        "n_epochs" : args.n_epochs,
        "batch_size" : args.batch_size,
    }

    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=args.experiment+datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        project="ai reward predictor training",
        entity="misoshiruseijin",
        config=train_config,
    )

    # train
    train(
        model=model,
        trainloader=trainloader, 
        train_features=train_features,
        test_features=test_features,
        train_rewards=train_rewards,
        test_rewards=test_rewards,
        n_epochs=args.n_epochs,
        lr=args.lr,
        model_save_dir=save_dir,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/hayano/all_aesthetic")
    parser.add_argument("--datafile", type=str, default="/home/hayano/all_aesthetic.pkl")
    parser.add_argument("--save-dir", type=str, default="/home/hayano/img_encoder_ai")
    parser.add_argument("--save-every", type=str, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[2048]*6)
    parser.add_argument("--n-hidden-layers", type=int, default=5)
    parser.add_argument("--output-dim", type=int, default=1)
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--featurefile", type=str, default="/home/hayano/all_aesthetic_feature.pt")
    parser.add_argument("--encoder-model-dir", type=str, default="/home/hayano/img_encoder_ai/test_ai_encoder")

    args = parser.parse_args()
    main(args)