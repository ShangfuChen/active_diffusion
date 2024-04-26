
import os
import time

import json
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader

import wandb

from rl4dgm.models.mydatasets import TripletDataset, DoubleTripletDataset
from rl4dgm.models.my_models import LinearModel

class TripletEncoderTrainer:
    """
    Class for keeping track of image encoder trained on triplet loss
    """
    def __init__(
            self, 
            config_dict: dict,
            device,
            trainset: TripletDataset = None,
            testset: TripletDataset = None, 
        ):
        """
        Args:
            model (nn.Module) : encoder model to train
            trainset and testset (TripletDataset) : datasets to use for training and testing. See TripletDataset class for more detail
            config_dict : 
                keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
        """
        default_config = {
            "batch_size" : 32,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 50,
            "triplet_margin" : 1.0,
            "save_dir" : None,
            "save_every" : 50,
            "input_dim" : 32768,
            "n_hidden_layers" : 5,
            "hidden_dims" : [22000]*6,
            "output_dim" : 4096,
        }

        # create directory to save config and model checkpoints 
        assert "save_dir" in config_dict.keys(), "config_dict is missing key: save_dir"
        os.makedirs(config_dict["save_dir"], exist_ok=False)
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        print("Initializing TripletEncoderTrainer with following configs\n", config_dict)
        with open(os.path.join(config_dict["save_dir"], "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved TripletEncoderTrainer config to", os.path.join(config_dict["save_dir"], "train_config.json"))
        
        self.device = device
        self.model = LinearModel(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            device=self.device,
        )
        self.trainset = trainset
        self.testset = testset
        self.config = config_dict

        # Initialize dataloaders
        self.dataloaders = {}
        self.initialize_dataloaders(trainset, testset)
        
        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = nn.TripletMarginLoss(p=2, margin=self.config["triplet_margin"])

        self.n_total_epochs = 0
        self.n_total_steps = 0

        self.start_time = time.time()

    def initialize_dataloaders(
            self, 
            trainset : TripletDataset = None, 
            testset : TripletDataset = None,
        ): 
        """
        Update dataset and reinitialize dataloaders
        Args:
            trainset (TripletDataset)
            testset (TripletDataset)
        """
        if trainset is not None:
            self.trainset = trainset
            self.trainloader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"])
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(testset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"])
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an image encoder using triplet loss
        """
        n_steps = 0
        for epoch in range(self.config["n_epochs"]):
            self.n_total_epochs += 1
            running_losses = []
            print("TripletEncoder training epoch", epoch)

            for step, (anchor_features, _, positive_features, negative_features) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                anchor_out = self.model(anchor_features)
                positive_out = self.model(positive_features)
                negative_out = self.model(negative_features)

                loss = self.criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                self.optimizer.step()
                running_losses.append(loss.item())
                n_steps += 1
                self.n_total_epochs += 1

                wandb.log({
                    "epoch" : self.n_total_epochs,
                    "step" : self.n_total_epochs,
                    "loss" : loss.item(),
                    "lr" : self.config["lr"],
                    "clock_time" : time.time() - self.start_time,
                })
            
            # save checkpoint
            if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                model_save_path = os.path.join(self.config["save_dir"], f"epoch{self.n_total_epochs}.pt")
                torch.save(self.model.state_dict(), model_save_path)
                print("TripletEncoder model checkpoint saved to", model_save_path)

    def eval_model(self):
        with torch.no_grad():
            for dataset_type, dataloader in self.dataloaders.item():
                print(f"TripletEncoder evaluation - {dataset_type}set")
                anchor_positive = []
                anchor_negative = []
                for step, (anchor_features, _, positive_features, negative_features) in enumerate(dataloader):
                    # get anchor-positive and anchor-negative distances
                    anchor_out = self.model(anchor_features)
                    positive_out = self.model(positive_features)
                    negative_out = self.model(negative_features)
                    anchor_positive_dist = torch.linalg.norm(anchor_out - positive_out, dim=1)
                    anchor_positive.append(anchor_positive_dist.mean().item())
                    anchor_negative_dist = torch.linalg.norm(anchor_out - negative_out, dim=1)
                    anchor_negative.append(anchor_negative_dist.mean().item())
                    
                wandb.log({
                    f"{dataset_type}_anchor_negative_dist" : np.array(anchor_negative).mean(),
                    f"{dataset_type}_anchor_positive_dist" : np.array(anchor_positive).mean(),
                    f"{dataset_type}_dist_diff" : (np.array(anchor_negative) - np.array(anchor_positive)).mean(),
                })

    
class DoubleTripletEncoderTrainer:
    """
    Class for keeping track of image encoder trained on weighted triplet loss using reward values from two different agents
    """
    def __init__(
            self, 
            config_dict: dict,
            trainset: DoubleTripletDataset = None,
            testset: DoubleTripletDataset = None, 
        ):
        """
        Args:
            model (nn.Module) : encoder model to train
            trainset and testset (DoubleTripletDataset) : datasets to use for training and testing. See TripletDataset class for more detail
            config_dict : 
                keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
        """
        default_config = {
            "batch_size" : 32,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 50,
            "save_dir" : None,
            "save_every" : 50,
            "agent1_triplet_margin" : 1.0,
            "agent2_triplet_margin" : 1.0,
            "agent1_loss_weight" : 1.0,
            "agent2_loss_weight" : 0.25,
            "input_dim" : 32768,
            "n_hidden_layers" : 5,
            "hidden_dims" : [22000]*6,
            "output_dim" : 4096,
        }

        # create directory to save config and model checkpoints 
        assert "save_dir" in config_dict.keys(), "config_dict is missing key: save_dir"
        os.makedirs(config_dict["save_dir"], exist_ok=False)
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        print("Initializing DoubleTripletEncoderTrainer with following configs\n", config_dict)
        with open(os.path.join(config_dict["save_dir"], "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved DoubleTripletEncoderTrainer config to", os.path.join(config_dict["save_dir"], "train_config.json"))
        
        self.model = LinearModel(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            device=self.device,
        )
        self.trainset = trainset
        self.testset = testset
        self.config = config_dict

        # Initialize dataloaders
        self.dataloaders = {}
        self.initialize_dataloaders(trainset, testset)
        
        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.agent1_criterion = nn.TripletMarginLoss(p=2, margin=self.config["agent1_triplet_margin"])
        self.agent2_criterion = nn.TripletMarginLoss(p=2, margin=self.config["agent2_triplet_margin"])

        self.n_total_epochs = 0
        self.n_total_steps = 0

        self.start_time = time.time()

    def initialize_dataloaders(
            self,
            trainset : DoubleTripletDataset = None, 
            testset : DoubleTripletDataset = None
        ):
        """
        Update dataset and reinitialize dataloaders
        Args:
            trainset (DoubleTripletDataset)
            testset (DoubleTripletDataset)
        """
        if trainset is not None:
            self.trainset = trainset
            self.trainloader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"])
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(testset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"])
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an image encoder using triplet loss
        """
        n_steps = 0
        
        for epoch in range(self.config["n_epochs"]):
            self.n_total_epochs += 1
            running_losses = []
            print("TripletEncoder training epoch", epoch)

            for step, (anchor_features, _, positive_feature_self, negative_feature_self, other_feature, is_positive) in enumerate(trainloader):
                self.optimizer.zero_grad()
                anchor_out = self.model(anchor_features)
                
                ############################################################
                # Compute loss coming from agent1 (self) rewards
                ############################################################
                positive_out_self = self.model(positive_feature_self)
                negative_out_self = self.model(negative_feature_self)
                loss_self = self.agent1_criterion(anchor_out, positive_out_self, negative_out_self)

                ############################################################
                # Compute loss coming from agent2 (other) rewards
                ############################################################
                positive_out_other = torch.zeros_like(anchor_out)
                positive_out_other[is_positive] = other_feature[is_positive]
                positive_out_other[~is_positive] = positive_out_self[~is_positive]
                negative_out_other = torch.zeros_like(anchor_out)
                negative_out_other[is_positive] = negative_out_self[is_positive]
                negative_out_other[~is_positive] = other_feature[~is_positive]
                loss_other = self.agent2_criterion(anchor_out, positive_out_other, negative_out_other)
                
                ############################################################
                # Compute overall loss
                ############################################################
                loss = (self.config["agent1_loss_weight"] * loss_self) + (self.config["agent2_loss_weight"] * loss_other) 

                # backprop and take a step
                loss.backward()
                self.optimizer.step()
                running_losses.append(loss.item())
                n_steps += 1
                self.n_total_steps += 1

                wandb.log({
                    "epoch" : self.n_total_epochs,
                    "step" : self.n_total_steps,
                    "loss_self" : loss_self.item(),
                    "loss_other" : loss_other.item(),
                    "loss" : loss.item(),
                    "lr" : self.config["lr"],
                    "clock_time" : time.time() - self.start_time,
                })

            # save checkpoint
            if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                model_save_path = os.path.join(self.config["save_dir"], f"epoch{self.n_total_epochs}.pt")
                torch.save(self.model.state_dict(), model_save_path)
                print("DoubleTripletEncoder model checkpoint saved to", model_save_path)

    def eval_model(self):
        with torch.no_grad():
            for dataset_type, dataloader in self.dataloaders.items():
                print(f"DoubleTripletEncoder evaluation - {dataset_type}set")

                anchor_positive_self = []
                anchor_negative_self = []
                anchor_positive_other = []
                anchor_negative_other = []

                for step, (anchor_features, _, positive_feature_self, negative_feature_self, other_feature, is_positive) in enumerate(dataloader):
                    anchor_out = self.model(anchor_features)
                    positive_out_self = self.model(positive_feature_self)
                    negative_out_self = self.model(negative_feature_self)
                    
                    anchor_positive_dist_self = torch.linalg.norm(anchor_out - positive_out_self, dim=1)
                    anchor_positive_self.append(anchor_positive_dist_self.mean().item())
                    anchor_negative_dist_self = torch.linalg.norm(anchor_out - negative_out_self, dim=1)
                    anchor_negative_self.append(anchor_negative_dist_self.mean().item())

                    anchor_positive_dist_other = torch.linalg.norm(anchor_out[is_positive] - other_feature[is_positive], dim=1)
                    anchor_positive_other.append(anchor_positive_dist_other.mean().item())
                    anchor_negative_dist_other = torch.linalg.norm(anchor_out[~is_positive] - other_feature[~is_positive], dim=1)
                    anchor_negative_other.append(anchor_negative_dist_other.mean().item())
                    
                wandb.log({
                    f"{dataset_type}_anchor_positive_dist_self" : np.array(anchor_positive_self).mean(),
                    f"{dataset_type}_anchor_negative_dist_self" : np.array(anchor_negative_self).mean(),
                    f"{dataset_type}_dist_diff_self" : (np.array(anchor_negative_self) - np.array(anchor_positive_self)).mean(),
                    f"{dataset_type}_anchor_positive_dist_other" : np.array(anchor_positive_other).mean(),
                    f"{dataset_type}_anchor_negative_dist_other" : np.array(anchor_negative_other).mean(),
                    f"{dataset_type}_dist_diff_other" : (np.array(anchor_negative_other) - np.array(anchor_positive_other)).mean(),
                })
  



            