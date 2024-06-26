
import os
import time

import json
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader

# from torchensemble import VotingRegressor

from accelerate import Accelerator

from rl4dgm.models.mydatasets import FeatureDoubleLabelDataset
from rl4dgm.models.my_models import LinearModel, MultiClassClassifierModel, LinearModelforEnsemble, EnsembleLinearModels

from omegaconf.listconfig import ListConfig

class ErrorPredictorTrainer:
    """
    Class for keeping track of feedback error prediction model and datasets
    """
    def __init__(
            self, 
            accelerator: Accelerator,
            seed,
            save_dir,
            trainset: FeatureDoubleLabelDataset,
            testset: FeatureDoubleLabelDataset, 
            config_dict: dict,
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
            "save_every" : 50,
            "n_hidden_layers" : 5,
            "hidden_dims" : [22000]*6,
            "output_dim" : 4096,
            "model_type" : "error_predictor",
            "loss_type" : "MSE",
            "name" : "error_predictor",
        }

        MODEL_TYPES = {
            "error_predictor" : LinearModel,
        }

        LOSS_TYPES = {
            "MSE" : torch.nn.MSELoss,
        }

        # create directory to save config and model checkpoints 
        os.makedirs(save_dir, exist_ok=False)
        self.save_dir = save_dir

        # make sure input dimension is defined in the config
        assert "input_dim" in config_dict.keys(), "config_dict is missing key: input_dim"
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        # hidden_dim is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
        config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]
        
        print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        self.accelerator = accelerator
        self.device = accelerator.device

        model_class = MODEL_TYPES[config_dict["model_type"]]
        self.model = model_class(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            device=self.device,
            # seed=seed,
        )

        self.trainset = trainset
        self.testset = testset
        self.config = config_dict
        self.name = config_dict["name"]

        # Initialize dataloaders
        self.dataloaders = {}
        self.datasets = {}
        self.initialize_dataloaders(trainset, testset)

        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        loss_function = LOSS_TYPES[config_dict["loss_type"]]
        self.criterion = loss_function()

        self.n_total_epochs = 0
        self.n_total_steps = 0
        self.n_calls_to_train = 0 # how many times the train function has been called

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
            self.trainloader = DataLoader(
                trainset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(
                testset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an error predictor
        """
        self.n_calls_to_train += 1
        n_steps = 0

        for epoch in range(self.config["n_epochs"]):
            self.n_total_epochs += 1
            running_losses = []
            if epoch % 100 == 0:
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

                self.accelerator.log({
                    f"{self.name}_epoch" : epoch,
                    f"{self.name}_step" : self.n_total_epochs,
                    f"{self.name}_loss" : loss.item(),
                    f"{self.name}_lr" : self.config["lr"],
                    f"{self.name}_clock_time" : time.time() - self.start_time,
                })
            
            # save checkpoint
            if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                model_save_path = os.path.join(self.save_dir, f"epoch{self.n_total_epochs}.pt")
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

                self.accelerator.log({
                    f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
                    f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
                    f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
                })

class RewardClassifierTrainer:
    """
    Class for keeping track of discrete reward classifier model and datasets
    """
    def __init__(
            self, 
            accelerator: Accelerator,
            seed,
            save_dir,
            trainset: FeatureDoubleLabelDataset,
            testset: FeatureDoubleLabelDataset, 
            config_dict: dict,
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
            "save_every" : 50,
            "n_hidden_layers" : 5,
            "hidden_dims" : [22000]*6,
            "output_dim" : 4096,
            "model_type" : "multiclass_classifier",
            "loss_type" : "CE",
            "name" : "multiclass_classifier",
        }

        MODEL_TYPES = {
            "multiclass_classifier" : MultiClassClassifierModel,
        }

        LOSS_TYPES = {
            "MSE" : torch.nn.MSELoss,
            # "CE" : torch.nn.CrossEntropyLoss,
        }

        # create directory to save config and model checkpoints 
        os.makedirs(save_dir, exist_ok=False)
        self.save_dir = save_dir

        # make sure input dimension is defined in the config
        assert "input_dim" in config_dict.keys(), "config_dict is missing key: input_dim"
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        # hidden_dim is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
        config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]
        
        print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        self.accelerator = accelerator
        self.device = accelerator.device

        model_class = MODEL_TYPES[config_dict["model_type"]]
        self.model = model_class(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            device=self.device,
            # seed=seed,
        )

        self.trainset = trainset
        self.testset = testset
        self.config = config_dict
        self.name = config_dict["name"]
        self.n_classes = config_dict["output_dim"]

        # Initialize dataloaders
        self.dataloaders = {}
        self.datasets = {}
        self.initialize_dataloaders(trainset, testset)

        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        loss_function = LOSS_TYPES[config_dict["loss_type"]]
        self.criterion = loss_function()

        self.n_total_epochs = 0
        self.n_total_steps = 0
        self.n_calls_to_train = 0 # how many times the train function has been called

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
            self.trainloader = DataLoader(
                trainset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(
                testset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an image encoder using triplet loss
        """
        n_steps = 0
        self.n_calls_to_train += 1

        for epoch in range(self.config["n_epochs"]):
            self.n_total_epochs += 1
            running_losses = []
            if epoch % 100 == 0:
                print("ErrorPredictor training epoch", epoch)

            for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                predictions = self.model(features) # prob
                # prepare ground truth labels from rewards
                gnd_truth_labels = torch.zeros([agent1_labels.shape[0], self.n_classes])
                for i, label in enumerate(agent1_labels):
                    gnd_truth_labels[i,:int(label.item())+1] = 1
                gnd_truth_labels = gnd_truth_labels.to(self.device)
                loss = self.criterion(predictions, gnd_truth_labels)

                loss.backward()
                self.optimizer.step()
                running_losses.append(loss.item())
                n_steps += 1
                self.n_total_steps += 1

                self.accelerator.log({
                    f"{self.name}_epoch" : epoch,
                    f"{self.name}_step" : self.n_total_steps,
                    f"{self.name}_loss" : loss.item(),
                    f"{self.name}_lr" : self.config["lr"],
                    f"{self.name}_clock_time" : time.time() - self.start_time,
                })
            
            # save checkpoint
            if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                model_save_path = os.path.join(self.save_dir, f"epoch{self.n_total_epochs}.pt")
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

                self.accelerator.log({
                    f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
                    f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
                    f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
                })

# class ErrorPredictorEnsembleTrainer:
#     """
#     Class for keeping track of feedback error prediction model (ensemble) and datasets
#     """
#     def __init__(
#             self, 
#             accelerator: Accelerator,
#             seed,
#             save_dir,
#             trainset: FeatureDoubleLabelDataset,
#             testset: FeatureDoubleLabelDataset, 
#             config_dict: dict,
#         ):
#         """
#         Args:
#             trainset and testset (FeatureDoubleLabelDataset) : datasets to use for training and testing. See TripletDataset class for more detail
#             config_dict : 
#                 keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
#         """
#         default_config = {
#             "batch_size" : 32,
#             "shuffle" : True,
#             "lr" : 1e-6,
#             "n_epochs" : 50,
#             "save_every" : 50,
#             "n_hidden_layers" : 5,
#             "hidden_dims" : [512]*4,
#             "output_dim" : 1,
#             "n_models" : 3, # number of models in ensemble
#             "model_type" : "linear",
#             "loss_type" : "MSE",
#             "name" : "error_predictor_ensemble",
#         }

#         # create directory to save config and model checkpoints 
#         os.makedirs(save_dir, exist_ok=False)
        
#         # make sure input dimension is defined in the config
#         assert "input_dim" in config_dict.keys(), "config_dict is missing key: input_dim"
            
#         # populate the config with default values if values are not provided
#         for key in default_config:
#             if key not in config_dict.keys():
#                 config_dict[key] = default_config[key]
#         # hidden_dim and model_seeds is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
#         config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]

#         print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
#         with open(os.path.join(save_dir, "train_config.json"), "w") as f:
#             json.dump(config_dict, f)
#             print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
#         self.seed = seed
#         self.generator = torch.Generator()
#         self.generator.manual_seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.cuda.manual_seed(seed)

#         self.accelerator = accelerator
#         self.device = accelerator.device

#         self.model = EnsembleLinearModels(
#             input_dim=config_dict["input_dim"],
#             hidden_dims=config_dict["hidden_dims"],
#             output_dim=config_dict["output_dim"],
#             n_models=config_dict["n_models"],
#             device=self.device,
#             # seed=seed,
#         )

#         self.trainset = trainset
#         self.testset = testset
#         self.config = config_dict
#         self.name = config_dict["name"]

#         # Initialize dataloaders
#         self.dataloaders = {}
#         self.datasets = {}
#         self.initialize_dataloaders(trainset, testset)

#         # Initialize optimizer and loss criteria
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
#         self.criterion = torch.nn.MSELoss()

#         self.n_total_epochs = 0
#         self.n_total_steps = 0
#         self.n_calls_to_train = 0 # how many times the train function has been called

#         self.start_time = time.time()

#     def get_emsemble_out_mean_std(self, x):
#         with torch.no_grad():
#             return self.model.get_all_model_outputs_mean_std(x)

#     def initialize_dataloaders(
#             self, 
#             trainset : FeatureDoubleLabelDataset = None, 
#             testset : FeatureDoubleLabelDataset = None,
#         ): 
#         """
#         Update dataset and reinitialize dataloaders
#         Args:
#             trainset (FeatureDoubleLabelDataset)
#             testset (FeatureDoubleLabelDataset)
#         """
#         if trainset is not None:
#             self.trainset = trainset
#             self.trainloader = DataLoader(
#                 trainset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["train"] = self.trainloader
#         if testset is not None:
#             self.testset = testset
#             self.testloader = DataLoader(
#                 testset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["test"] = self.testloader

#     def train_model(self):
#         """
#         Trains an error predictor
#         """
#         self.n_calls_to_train += 1
#         n_steps = 0

#         for epoch in range(self.config["n_epochs"]):
#             self.n_total_epochs += 1
#             running_losses = []
#             if epoch % 100 == 0:
#                 print("ErrorPredictor training epoch", epoch)

#             for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
#                 self.optimizer.zero_grad()
#                 predictions = torch.flatten(self.model(features))
#                 true_error = agent1_labels - agent2_labels
#                 loss = self.criterion(predictions, true_error)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_losses.append(loss.item())
#                 n_steps += 1
#                 self.n_total_steps += 1

#                 self.accelerator.log({
#                     f"{self.name}_epoch" : epoch,
#                     f"{self.name}_step" : self.n_total_epochs,
#                     f"{self.name}_loss" : loss.item(),
#                     f"{self.name}_lr" : self.config["lr"],
#                     f"{self.name}_clock_time" : time.time() - self.start_time,
#                 })
            
#             # save checkpoint
#             if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
#                 model_save_path = os.path.join(save_dir, f"epoch{self.n_total_epochs}.pt")
#                 torch.save(self.model.state_dict(), model_save_path)
#                 print("ErrorPredictor model checkpoint saved to", model_save_path)

#     def eval_model(self):
#         with torch.no_grad():
#             for dataset_type, dataloader in self.dataloaders.item():
#                 print(f"ErrorPredictor evaluation - {dataset_type}set")

#                 prediction_errors = []
#                 for step, (features, agent1_labels, agent2_labels) in enumerate(dataloader):
#                     predicted_error = self.model(features)
#                     predicted_agent1_labels = agent2_labels + predicted_error # apply correction using predicted error
#                     agent1_label_prediction_error = torch.abs(agent1_labels - predicted_agent1_labels)
#                     prediction_errors.append(agent1_label_prediction_error)

#                 prediction_errors = np.array(prediction_errors)
#                 prediction_percent_errors = prediction_errors / (self.datasets[dataset_type].agent1_label_max - self.datasets[dataset_type].agent1_label_min) 

#                 self.accelerator.log({
#                     f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
#                     f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
#                     f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
#                 })

# class ErrorPredictorEnsembleTrainer2:
#     """
#     version with ensembletorch
#     Class for keeping track of feedback error prediction model (ensemble) and datasets
#     """
#     def __init__(
#             self, 
#             accelerator: Accelerator,
#             seed,
#             save_dir,
#             trainset: FeatureDoubleLabelDataset,
#             testset: FeatureDoubleLabelDataset, 
#             config_dict: dict,
#         ):
#         """
#         Args:
#             trainset and testset (FeatureDoubleLabelDataset) : datasets to use for training and testing. See TripletDataset class for more detail
#             config_dict : 
#                 keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
#         """
#         # create directory to save config and model checkpoints 
#         os.makedirs(save_dir, exist_ok=False)
#         for key, value in config_dict.items():
#             if isinstance(value, ListConfig):
#                 config_dict[key] = [item for item in value]

#         print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
#         with open(os.path.join(save_dir, "train_config.json"), "w") as f:
#             json.dump(config_dict, f)
#             print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
#         self.seed = seed
#         self.generator = torch.Generator()
#         self.generator.manual_seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.cuda.manual_seed(seed)

#         self.accelerator = accelerator
#         self.device = accelerator.device

#         self.model = VotingRegressor(
#             estimator=LinearModelforEnsemble,
#             n_estimators=10,
#             cuda=True,
#         )
#         self.model.set_criterion(torch.nn.MSELoss())
#         self.model.set_optimizer(
#             optimizer_name="Adam",
#             lr=config_dict["lr"],
#         )
#         print("...done\n")

#         self.trainset = trainset
#         self.testset = testset
#         self.config = config_dict
#         self.name = config_dict["name"]

#         # Initialize dataloaders
#         self.dataloaders = {}
#         self.datasets = {}
#         self.initialize_dataloaders(trainset, testset)

#         # Initialize optimizer and loss criteria
#         # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
#         # self.criterion = torch.nn.MSELoss()

#         self.n_total_epochs = 0
#         self.n_total_steps = 0
#         self.n_calls_to_train = 0 # how many times the train function has been called

#         self.start_time = time.time()

#     def get_emsemble_out_mean_std(self, x):
#         with torch.no_grad():
#             return self.model.get_all_model_outputs_mean_std(x)

#     def initialize_dataloaders(
#             self, 
#             trainset : FeatureDoubleLabelDataset = None, 
#             testset : FeatureDoubleLabelDataset = None,
#         ): 
#         """
#         Update dataset and reinitialize dataloaders
#         Args:
#             trainset (FeatureDoubleLabelDataset)
#             testset (FeatureDoubleLabelDataset)
#         """
#         if trainset is not None:
#             self.trainset = trainset
#             self.trainloader = DataLoader(
#                 trainset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["train"] = self.trainloader
#         if testset is not None:
#             self.testset = testset
#             self.testloader = DataLoader(
#                 testset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["test"] = self.testloader

#     def train_model(self):
#         """
#         Trains an error predictor
#         """
#         self.n_calls_to_train += 1

#         self.model.fit(
#             train_loader=self.trainloader,
#             epochs=self.config["n_epochs"]
#         )
#         # n_steps = 0

#         # for epoch in range(self.config["n_epochs"]):
#         #     self.n_total_epochs += 1
#         #     running_losses = []
#         #     if epoch % 100 == 0:
#         #         print("ErrorPredictor training epoch", epoch)

#         #     for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
#         #         self.optimizer.zero_grad()
#         #         predictions = torch.flatten(self.model(features))
#         #         true_error = agent1_labels - agent2_labels
#         #         loss = self.criterion(predictions, true_error)
#         #         loss.backward()
#         #         self.optimizer.step()
#         #         running_losses.append(loss.item())
#         #         n_steps += 1
#         #         self.n_total_steps += 1

#         #         self.accelerator.log({
#         #             f"{self.name}_epoch" : epoch,
#         #             f"{self.name}_step" : self.n_total_epochs,
#         #             f"{self.name}_loss" : loss.item(),
#         #             f"{self.name}_lr" : self.config["lr"],
#         #             f"{self.name}_clock_time" : time.time() - self.start_time,
#         #         })
            
#         #     # save checkpoint
#         #     if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
#         #         model_save_path = os.path.join(save_dir, f"epoch{self.n_total_epochs}.pt")
#         #         torch.save(self.model.state_dict(), model_save_path)
#         #         print("ErrorPredictor model checkpoint saved to", model_save_path)

#     def eval_model(self):
#         with torch.no_grad():
#             for dataset_type, dataloader in self.dataloaders.item():
#                 print(f"ErrorPredictor evaluation - {dataset_type}set")

#                 prediction_errors = []
#                 for step, (features, agent1_labels, agent2_labels) in enumerate(dataloader):
#                     predicted_error = self.model(features)
#                     predicted_agent1_labels = agent2_labels + predicted_error # apply correction using predicted error
#                     agent1_label_prediction_error = torch.abs(agent1_labels - predicted_agent1_labels)
#                     prediction_errors.append(agent1_label_prediction_error)

#                 prediction_errors = np.array(prediction_errors)
#                 prediction_percent_errors = prediction_errors / (self.datasets[dataset_type].agent1_label_max - self.datasets[dataset_type].agent1_label_min) 

#                 self.accelerator.log({
#                     f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
#                     f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
#                     f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
#                 })

class ErrorPredictorEnsembleTrainerVoting:
    """
    Class for keeping track of feedback error prediction model (ensemble) and datasets
    """
    def __init__(
            self, 
            accelerator: Accelerator,
            seed,
            save_dir,
            trainset: FeatureDoubleLabelDataset,
            testset: FeatureDoubleLabelDataset, 
            config_dict: dict,
        ):
        """
        Args:
            trainset and testset (FeatureDoubleLabelDataset) : datasets to use for training and testing. See TripletDataset class for more detail
            config_dict : 
                keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
        """
        default_config = {
            "batch_size" : 32,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 50,
            "save_every" : 50,
            "n_hidden_layers" : 5,
            "hidden_dims" : [512]*4,
            "output_dim" : 1,
            "n_models" : 3, # number of models in ensemble
            "model_type" : "linear",
            "loss_type" : "MSE",
            "name" : "error_predictor_ensemble",
        }

        # create directory to save config and model checkpoints 
        # for i in range(config_dict["n_models"]):
        #     os.makedirs(os.path.join(save_dir, f"model{i}"), exist_ok=False)
        os.makedirs(os.path.join(save_dir, f"model0"), exist_ok=True)
        self.save_dir = save_dir

        # make sure input dimension is defined in the config
        assert "input_dim" in config_dict.keys(), "config_dict is missing key: input_dim"
            
        # populate the config with default values if values are not provided
        for key in default_config:
            if key not in config_dict.keys():
                config_dict[key] = default_config[key]
        # hidden_dim and model_seeds is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
        config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]

        print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(config_dict, f)
            print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        self.accelerator = accelerator
        self.device = accelerator.device

        self.model = EnsembleLinearModels(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            output_dim=config_dict["output_dim"],
            n_models=config_dict["n_models"],
            device=self.device,
            # seed=seed,
        )

        self.trainset = trainset
        self.testset = testset
        self.config = config_dict
        self.name = config_dict["name"]

        # Initialize dataloaders
        self.dataloaders = {}
        self.datasets = {}
        self.initialize_dataloaders(trainset, testset)

        # Initialize optimizer and loss criteria
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.criterion = torch.nn.MSELoss()

        self.n_total_epochs = 0
        self.n_total_steps = 0
        self.n_calls_to_train = 0 # how many times the train function has been called

        self.start_time = time.time()

    def get_emsemble_out_mean_std(self, x):
        with torch.no_grad():
            return self.model.get_all_model_outputs_mean_std(x)

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
            self.trainloader = DataLoader(
                trainset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["train"] = self.trainloader
        if testset is not None:
            self.testset = testset
            self.testloader = DataLoader(
                testset, 
                batch_size=self.config["batch_size"], 
                shuffle=self.config["shuffle"],
                generator=self.generator,
            )
            self.dataloaders["test"] = self.testloader

    def train_model(self):
        """
        Trains an error predictor
        """
        self.n_calls_to_train += 1
        n_steps = 0

        for i, model in enumerate(self.model.models):
            print(f"Training model {i}/{len(self.model.models)}")
            for epoch in range(self.config["n_epochs"]):
                if i == 0:
                    self.n_total_epochs += 1
                running_losses = []
                if epoch % 500 == 0:
                    print("ErrorPredictor training epoch", epoch)

                for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
                    self.optimizer.zero_grad()
                    predictions = torch.flatten(model(features))
                    true_error = agent1_labels - agent2_labels
                    loss = self.criterion(predictions, true_error)
                    loss.backward()
                    self.optimizer.step()
                    running_losses.append(loss.item())
                    n_steps += 1
                    self.n_total_steps += 1

                    self.accelerator.log({
                        # f"{self.name}_epoch{i}" : epoch,
                        # f"{self.name}_step{i}" : self.n_total_epochs,
                        f"{self.name}_loss{i}" : loss.item(),
                        # f"{self.name}_model{i}_lr" : self.config["lr"],
                        # f"{self.name}_model{i}_clock_time" : time.time() - self.start_time,
                    })
                
                # save checkpoint
                if i == 0 and (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                    model_save_path = os.path.join(self.save_dir, f"model0", f"epoch{self.n_total_epochs}.pt")
                    torch.save(model.state_dict(), model_save_path)
                    print("ErrorPredictor model checkpoint saved to", model_save_path)
                # # save checkpoint
                # if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
                #     model_save_path = os.path.join(self.save_dir, f"model{i}", f"epoch{self.n_total_epochs}.pt")
                #     torch.save(model.state_dict(), model_save_path)
                #     print("ErrorPredictor model checkpoint saved to", model_save_path)


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

                self.accelerator.log({
                    f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
                    f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
                    f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
                    f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
                })


# For softmax output model
# class RewardClassifierTrainer:
#     """
#     Class for keeping track of discrete reward classifier model and datasets
#     """
#     def __init__(
#             self, 
#             accelerator: Accelerator,
#             seed,
#             trainset: FeatureDoubleLabelDataset,
#             testset: FeatureDoubleLabelDataset, 
#             config_dict: dict,
#         ):
#         """
#         Args:
#             model (nn.Module) : error predictor model to train
#             trainset and testset (FeatureDoubleLabelDataset) : datasets to use for training and testing. See TripletDataset class for more detail
#             config_dict : 
#                 keys: batch_size, shuffle, lr, n_epochs, triplet_margin, save_dir, save_every
#         """
#         default_config = {
#             "batch_size" : 32,
#             "shuffle" : True,
#             "lr" : 1e-6,
#             "n_epochs" : 50,
#             "save_dir" : None,
#             "save_every" : 50,
#             "n_hidden_layers" : 5,
#             "hidden_dims" : [22000]*6,
#             "output_dim" : 4096,
#             "model_type" : "multiclass_classifier",
#             "loss_type" : "CE",
#             "name" : "multiclass_classifier",
#         }

#         MODEL_TYPES = {
#             "multiclass_classifier" : MultiClassClassifierModel,
#         }

#         LOSS_TYPES = {
#             "MSE" : torch.nn.MSELoss,
#             "CE" : torch.nn.CrossEntropyLoss,
#         }

#         # create directory to save config and model checkpoints 
#         assert "save_dir" in config_dict.keys(), "config_dict is missing key: save_dir"
#         os.makedirs(save_dir, exist_ok=False)
        
#         # make sure input dimension is defined in the config
#         assert "input_dim" in config_dict.keys(), "config_dict is missing key: input_dim"
            
#         # populate the config with default values if values are not provided
#         for key in default_config:
#             if key not in config_dict.keys():
#                 config_dict[key] = default_config[key]
#         # hidden_dim is ListConfig type if speficied in hydra config. Convert to list so it can be dumped to json
#         config_dict["hidden_dims"] = [dim for dim in config_dict["hidden_dims"]]
        
#         print("Initializing ErrorPredictorTrainer with following configs\n", config_dict)
#         with open(os.path.join(save_dir, "train_config.json"), "w") as f:
#             json.dump(config_dict, f)
#             print("saved ErrorPredictorTrainer config to", os.path.join(save_dir, "train_config.json"))
        
#         self.seed = seed
#         self.generator = torch.Generator()
#         self.generator.manual_seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.cuda.manual_seed(seed)

#         self.accelerator = accelerator
#         self.device = accelerator.device

#         model_class = MODEL_TYPES[config_dict["model_type"]]
#         self.model = model_class(
#             input_dim=config_dict["input_dim"],
#             hidden_dims=config_dict["hidden_dims"],
#             output_dim=config_dict["output_dim"],
#             device=self.device,
#             seed=seed,
#         )

#         self.trainset = trainset
#         self.testset = testset
#         self.config = config_dict
#         self.name = config_dict["name"]

#         # Initialize dataloaders
#         self.dataloaders = {}
#         self.datasets = {}
#         self.initialize_dataloaders(trainset, testset)

#         # Initialize optimizer and loss criteria
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
#         loss_function = LOSS_TYPES[config_dict["loss_type"]]
#         self.criterion = loss_function()

#         self.n_total_epochs = 0
#         self.n_total_steps = 0
#         self.n_calls_to_train = 0 # how many times the train function has been called

#         self.start_time = time.time()

#     def initialize_dataloaders(
#             self, 
#             trainset : FeatureDoubleLabelDataset = None, 
#             testset : FeatureDoubleLabelDataset = None,
#         ): 
#         """
#         Update dataset and reinitialize dataloaders
#         Args:
#             trainset (FeatureDoubleLabelDataset)
#             testset (FeatureDoubleLabelDataset)
#         """
#         if trainset is not None:
#             self.trainset = trainset
#             self.trainloader = DataLoader(
#                 trainset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["train"] = self.trainloader
#         if testset is not None:
#             self.testset = testset
#             self.testloader = DataLoader(
#                 testset, 
#                 batch_size=self.config["batch_size"], 
#                 shuffle=self.config["shuffle"],
#                 generator=self.generator,
#             )
#             self.dataloaders["test"] = self.testloader

#     def train_model(self):
#         """
#         Trains an image encoder using triplet loss
#         """
#         n_steps = 0
#         self.n_calls_to_train += 1

#         for epoch in range(self.config["n_epochs"]):
#             self.n_total_epochs += 1
#             running_losses = []
#             if epoch % 100 == 0:
#                 print("ErrorPredictor training epoch", epoch)

#             for step, (features, agent1_labels, agent2_labels) in enumerate(self.trainloader):
#                 self.optimizer.zero_grad()
#                 predictions = self.model(features) # prob
#                 # print("true labels", agent1_labels)
#                 # print("predicted labels", predictions)
#                 if self.config["loss_type"] == "CE":
#                     loss = self.criterion(predictions, agent1_labels.long())

#                 elif self.config["loss_type"] == "MSE":
#                     # convert probabilities to score label and apply MSE loss
#                     predicted_scores = torch.argmax(predictions, dim=1)
#                     loss = self.criterion(predicted_scores, agent1_labels)

#                 breakpoint()
#                 loss.backward()
#                 self.optimizer.step()
#                 running_losses.append(loss.item())
#                 n_steps += 1
#                 self.n_total_steps += 1

#                 self.accelerator.log({
#                     f"{self.name}_epoch" : epoch,
#                     f"{self.name}_step" : self.n_total_steps,
#                     f"{self.name}_loss" : loss.item(),
#                     f"{self.name}_lr" : self.config["lr"],
#                     f"{self.name}_clock_time" : time.time() - self.start_time,
#                 })
            
#             # save checkpoint
#             if (self.n_total_epochs > 0) and (self.n_total_epochs % self.config["save_every"]) == 0:
#                 model_save_path = os.path.join(save_dir, f"epoch{self.n_total_epochs}.pt")
#                 torch.save(self.model.state_dict(), model_save_path)
#                 print("ErrorPredictor model checkpoint saved to", model_save_path)

#     def eval_model(self):
#         with torch.no_grad():
#             for dataset_type, dataloader in self.dataloaders.item():
#                 print(f"ErrorPredictor evaluation - {dataset_type}set")

#                 prediction_errors = []
#                 for step, (features, agent1_labels, agent2_labels) in enumerate(dataloader):
#                     predicted_error = self.model(features)
#                     predicted_agent1_labels = agent2_labels + predicted_error # apply correction using predicted error
#                     agent1_label_prediction_error = torch.abs(agent1_labels - predicted_agent1_labels)
#                     prediction_errors.append(agent1_label_prediction_error)

#                 prediction_errors = np.array(prediction_errors)
#                 prediction_percent_errors = prediction_errors / (self.datasets[dataset_type].agent1_label_max - self.datasets[dataset_type].agent1_label_min) 

#                 self.accelerator.log({
#                     f"{self.name}_{dataset_type}_mean_error" : prediction_errors.mean(),
#                     f"{self.name}_{dataset_type}_mean_percent_error" : prediction_percent_errors.mean(),
#                     f"{self.name}_{dataset_type}_samples_with_under_10%_error" : (prediction_percent_errors < 0.1).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_20%_error" : (prediction_percent_errors < 0.2).sum() / predicted_agent1_labels.shape[0],
#                     f"{self.name}_{dataset_type}_samples_with_under_30%_error" : (prediction_percent_errors < 0.3).sum() / predicted_agent1_labels.shape[0],
#                 })
