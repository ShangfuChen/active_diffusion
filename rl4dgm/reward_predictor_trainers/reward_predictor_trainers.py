
import os
import time

import json
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader

import wandb

from rl4dgm.models.mydatasets import FeatureDoubleLabelDataset
from rl4dgm.models.my_models import LinearModel

class ErrorPredictorTrainer:
    """
    Class for keeping track of feedback error prediction model and datasets
    """
    def __init__(
            self, 
            trainset: FeatureDoubleLabelDataset,
            testset: FeatureDoubleLabelDataset, 
            config_dict: dict
        ):
        """
        Args:
            model (nn.Module) : error predictor model to train
            trainset and testset (FeatureDoubleLabelDataset) : datasets to use for training and testing. See TripletDataset class for more detail
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
        print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
        with open(os.path.join(config_dict["save_dir"], "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved ErrorPredictorTrainer config to", os.path.join(config_dict["save_dir"], "train_config.json"))
        
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
        self.datasets = {}
        self.initialize_dataloaders(trainset, testset)

        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = nn.MSELoss()

        self.n_total_epochs = 0
        self.n_total_steps = 0

        self.start_time = time.time()

    def initialize_dataloaders(
            self, 
            trainset : FeatureDoubleLabelDataset = None, 
            testset : FeatureDoubleLabelDataset = None,
        ): 
        """
        Update dataset and reinitialize dataloaders
        Args:
            trainset (FeatureDoubleLabelDataset)
            testset (FeatureDoubleLabelDataset)
        """
        if trainset is not None:
            self.trainset = trainset
            self.datasets["train"] = trainset
            self.trainloader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"])
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.datasets["test"] = testset
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
            print("ErrorPredictor training epoch", epoch)

            for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                predictions = torch.flatten(self.model(features))
                true_error = agent1_labels - agent2_labels
                loss = self.criterion(predictions, true_error)
                loss.backward()
                self.optimizer.step()
                running_losses.append(loss.item())
                n_steps += 1
                self.n_total_steps += 1

                wandb.log({
                    "epoch" : epoch,
                    "step" : n_steps,
                    "loss" : loss.item(),
                    "lr" : self.config["lr"],
                    "clock_time" : time.time() - self.start_time,
                })
            
            # save checkpoint
            if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                model_save_path = os.path.join(self.config["save_dir"], f"epoch{self.n_total_epochs}.pt")
                torch.save(self.model.state_dict(), model_save_path)
                print("ErrorPredictor model checkpoint saved to", model_save_path)

    def eval_model(self):
        with torch.no_grad():
            for dataset_type, dataloader in self.dataloaders.item():
                print(f"ErrorPredictor evaluation - {dataset_type}set")

                prediction_errors = []
                for step, (features, agent1_labels, agent2_labels) in enumerate(dataloader):
                    predicted_error = self.model(features)
                    predicted_agent1_labels = agent2_labels + predicted_error # apply correction using predicted error
                    agent1_label_prediction_error = torch.abs(agent1_labels - predicted_agent1_labels)
                    prediction_errors.append(agent1_label_prediction_error)

                prediction_errors = np.array(prediction_errors)
                prediction_percent_errors = prediction_errors / (self.datasets[dataset_type].agent1_label_max - self.datasets[dataset_type].agent1_label_min) 

                wandb.log({
                    f"{dataset_type}_mean_error" : prediction_errors.mean(),
                    f"{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
                    f"{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
                    f"{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
                    f"{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
                })
