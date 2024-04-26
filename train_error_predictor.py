
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
from rl4dgm.models.mydatasets import FeatureLabelDataset


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

def get_datasets(human_features, ai_features, human_rewards, ai_rewards, n_samples_per_epoch, train_ratio, batch_size, device):
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

    # split into train and test sets
    assert human_features.shape[0] % n_samples_per_epoch == 0, "number of features is not a multiple of n_samples_per_epoch"
    n_epochs = int(human_features.shape[0] / n_samples_per_epoch)
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

    # concatenate human and ai features
    train_features = torch.cat((human_features[train_indices], ai_features[train_indices]), dim=1)
    test_features = torch.cat((human_features[test_indices], ai_features[test_indices]), dim=1)
    
    train_human_rewards = torch.tensor(human_rewards[train_indices].values)
    test_human_rewards = torch.tensor(human_rewards[test_indices].values)
    train_ai_rewards = torch.tensor(ai_rewards[train_indices].values)
    test_ai_rewards = torch.tensor(ai_rewards[test_indices].values)

    train_error = train_human_rewards - train_ai_rewards
    test_error = test_human_rewards - test_ai_rewards

    print("Split into train and test")
    print("Initializing train set")
    trainset = FeatureLabelDataset(
        features=train_features, 
        labels=train_error,
        device=device,
    )
    print("Initializing test set")
    testset = FeatureLabelDataset(
        features=test_features,
        labels=test_error,
        device=device,
    )

    print("Initializing DataLoaders")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader, train_features, test_features, train_human_rewards, test_human_rewards, train_ai_rewards, test_ai_rewards

def train(
    model, trainloader, testloader, 
    train_features, test_features, 
    train_human_labels, test_human_labels, 
    train_ai_labels, test_ai_labels, 
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
        for step, (features, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            predictions = torch.flatten(model(features))
            # print("predicted error", predictions)
            # print("true error", labels)
            loss = criterion(predictions, labels)
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
            train_outputs = torch.flatten(model(train_features))
            test_outputs = torch.flatten(model(test_features))

            # apply correction to ai rewards
            train_human_rewards_prediction = train_ai_labels + train_outputs
            test_human_rewards_prediction = test_ai_labels + test_outputs

            # compute error
            train_human_errors = torch.abs(train_human_rewards_prediction - train_human_labels)
            test_human_errors = torch.abs(test_human_rewards_prediction - test_human_labels)

            # compute percent error
            train_percent_error = train_human_errors / (train_human_labels.max() - train_human_labels.min())
            test_percent_error = test_human_errors / (test_human_labels.max() - test_human_labels.min())

            # breakpoint()

            wandb.log({
                "train_mean_error" : train_human_errors.mean(),
                "train_mean_percent_error" : train_percent_error.mean(),
                "train_samples_with_under_10%_error" : (train_percent_error < 0.1).sum() / train_outputs.shape[0],
                "train_samples_with_under_20%_error" : (train_percent_error < 0.2).sum() / train_outputs.shape[0],
                "train_samples_with_under_30%_error" : (train_percent_error < 0.3).sum() / train_outputs.shape[0],
                # "train_samples_with_under_40%_error" : (train_percent_error < 0.4).sum() / train_outputs.shape[0],
                # "train_samples_with_under_50%_error" : (train_percent_error < 0.5).sum() / train_outputs.shape[0],
                "test_mean_error" : test_human_errors.mean(),
                "test_mean_percent_error" : test_percent_error.mean(),
                "test_samples_with_under_10%_error" : (test_percent_error < 0.1).sum() / test_outputs.shape[0],
                "test_samples_with_under_20%_error" : (test_percent_error < 0.2).sum() / test_outputs.shape[0],
                "test_samples_with_under_30%_error" : (test_percent_error < 0.3).sum() / test_outputs.shape[0],
                # "test_samples_with_under_40%_error" : (test_percent_error < 0.4).sum() / test_outputs.shape[0],
                # "test_samples_with_under_50%_error" : (test_percent_error < 0.5).sum() / test_outputs.shape[0],
            })
            
def main(args):

    # set seed
    torch.manual_seed(0)

    # experiment time
    experiment_time = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    # create directory to save model 
    save_dir = os.path.join(args.save_dir, f"{args.agent}_{args.experiment}", experiment_time)
    os.makedirs(save_dir, exist_ok=False)

    #############################################################################################
    # prepare SD features
    #############################################################################################
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

    #############################################################################################
    # get human and ai labels from dataframe
    #############################################################################################
    df = pd.read_pickle(args.datafile)
    human_rewards = df["human_rewards"]
    ai_rewards = df["ai_rewards"]

    #############################################################################################
    # load pretrained human encoder and encode SD features
    #############################################################################################
    with open(args.human_encoder_conf) as f:
        human_encoder_conf = json.load(f)
    human_encoder_model = LinearModel(
        input_dim=human_encoder_conf["input_dim"],
        hidden_dims=human_encoder_conf["hidden_dims"],
        output_dim=human_encoder_conf["output_dim"],
        device=args.device,
    )
    human_encoder_model.load_state_dict(torch.load(args.human_encoder_state_dict, map_location=torch.device("cpu")))
    human_encoder_model.to(args.device)
    print(human_encoder_model)
    print("loaded human encoder model")

    # encode features via human encoder
    with torch.no_grad():
        human_features = human_encoder_model(features.to(args.device)).cpu()
        print("encoded features using human encoder. latent shape", human_features.shape)
    
    # this model is no longer needed. free gpu memory
    del human_encoder_model

    #############################################################################################
    # load pretrained ai encoder and encode SD features
    #############################################################################################
    with open(args.ai_encoder_conf) as f:
        ai_encoder_conf = json.load(f)
    ai_encoder_model = LinearModel(
        input_dim=ai_encoder_conf["input_dim"],
        hidden_dims=ai_encoder_conf["hidden_dims"],
        output_dim=ai_encoder_conf["output_dim"],
        device=args.device,
    )
    ai_encoder_model.load_state_dict(torch.load(args.ai_encoder_state_dict, map_location=torch.device("cpu")))
    ai_encoder_model.to(args.device)
    print(ai_encoder_model)
    print("loaded ai encoder model")

    # encode features via human encoder
    with torch.no_grad():
        ai_features = ai_encoder_model(features.to(args.device)).cpu()
        print("encoded features using ai encoder. latent shape", ai_features.shape)
    
    # this model is no longer needed. free gpu memory
    del ai_encoder_model

    #############################################################################################
    # prepare dataset
    #############################################################################################
    trainloader, testloader, train_features, test_features, train_human_rewards, test_human_rewards, train_ai_rewards, test_ai_rewards = get_datasets(
        human_features=human_features,
        ai_features=ai_features,
        human_rewards=human_rewards,
        ai_rewards=ai_rewards,
        n_samples_per_epoch=args.n_samples_per_epoch,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("initialized dataloaders")

    #############################################################################################
    # initialize reward error prediction model
    #############################################################################################
    error_prediction_model = LinearModel(
        input_dim=train_features.shape[1],
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        device=args.device,
    )
    error_prediction_model.to(args.device)
    print(error_prediction_model)
    print("initialized reward errpr prediction model")

    #############################################################################################
    # initialize wandb for logging
    ############################################################################################## setup wandb for logging
    wandb.init(
        # name=datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
        name=args.experiment+experiment_time,
        project="reward_error_predictor_training",
        entity="misoshiruseijin",
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
        model=error_prediction_model,
        trainloader=trainloader,
        testloader=testloader,
        train_features=train_features.to(args.device),
        test_features=test_features.to(args.device),
        train_human_labels=train_human_rewards.to(args.device),
        test_human_labels=test_human_rewards.to(args.device),
        train_ai_labels=train_ai_rewards.to(args.device),
        test_ai_labels=test_ai_rewards.to(args.device),
        n_epochs=args.n_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/ayanoh/all_aesthetic")
    parser.add_argument("--datafile", type=str, default="/home/ayanoh/all_aesthetic.pkl")
    parser.add_argument("--save-dir", type=str, default="/home/ayanoh/reward_model_training_results")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples-per-epoch", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[22000]*6)
    parser.add_argument("--n-hidden-layers", type=int, default=5)
    parser.add_argument("--use-ai-feedback", action="store_true")
    parser.add_argument("--ai-feedback-dim", type=int, default=2048)
    parser.add_argument("--experiment", type=str, default="reward_error_prediction")
    parser.add_argument("--agent", type=str, default="ai", help="which agent's rewards to use for encoder training - ai or human")
    parser.add_argument("--human-encoder-state-dict", type=str, help="path to human encoder state dict pt file to load")    
    parser.add_argument("--human-encoder-conf", type=str, help="path to human encoder initialization config json file")
    parser.add_argument("--featurefile", type=str, default="/home/hayano/all_aesthetic_feature.pt")
    parser.add_argument("--ai-encoder-conf", type=str, help="path to ai encoder initialization config json file")
    parser.add_argument("--ai-encoder-state-dict", type=str, help="path to ai encoder state dict pt file to load")
    parser.add_argument("--output-dim", type=int, default=1)

    args = parser.parse_args()
    main(args)