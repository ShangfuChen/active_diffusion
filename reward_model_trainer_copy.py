
import json
import os
from typing import Any
import shutil
import datetime

import hydra
from hydra.utils import instantiate

import torch
from torch import nn
from torchvision.transforms.functional import to_tensor, to_pil_image

from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

from diffusers import DiffusionPipeline

from transformers import AutoProcessor, AutoModel

from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import QueryGenerator
from rl4dgm.utils.create_dummy_dataset import preference_from_ranked_prompts, preference_from_keyphrases

from rl4dgm.utils.generate_images import ImageGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from accelerate import Accelerator

class PickScoreTrainer:
    def __init__(self, cfg : DictConfig, logger):
        
        self.accelerator = instantiate_with_cfg(cfg.accelerator)

        if cfg.debug.activate and self.accelerator.is_main_process:
            import pydevd_pycharm
            pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

        if self.accelerator.is_main_process:
            self.verify_or_write_config(cfg, logger)

        logger.info(f"Loading task")
        self.task = self.load_task(cfg.task, self.accelerator)
        
        logger.info(f"Loading model")
        self.model = instantiate_with_cfg(cfg.model)
        
        logger.info(f"Loading criterion")
        self.criterion = instantiate_with_cfg(cfg.criterion)
        
        logger.info(f"Loading optimizer")
        self.optimizer = self.load_optimizer(cfg.optimizer, self.model)
        
        logger.info(f"Loading lr scheduler")
        self.lr_scheduler = self.load_scheduler(cfg.lr_scheduler, self.optimizer)
        breakpoint()
        validloader = self.load_dataloaders(cfg.dataset, split="validation")
        self.model, self.optimizer, self.lr_scheduler, validloader= self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, validloader)
        breakpoint()
        #####################
        # split2dataloader = self.load_dataloaders(cfg.dataset)
        # dataloaders = list(split2dataloader.values())
        # self.model, self.optimizer, self.lr_scheduler, *dataloaders = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler, *dataloaders)
        # self.split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))
        # self.dataloaders = dataloaders
        #####################

        self.accelerator.load_state_if_needed()

        # self.accelerator.recalc_train_length_after_prepare(len(split2dataloader[cfg.dataset.train_split_name]))

        self.accelerator.init_training(cfg)

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)

        # initialize feedback interface - TODO get type from confg and initialize appropriate class
        # self.feedback_interface = AIFeedbackInterface(preference_function=preference_from_keyphrases)
        self.feedback_interface = HumanFeedbackInterface()

        # initialize query generator
        self.query_generator = QueryGenerator()

        # Create files to save images and dataset to
        self.image_save_path = os.path.join(cfg.output_dir, "image_data")
        date_and_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dataset_save_path = os.path.join(cfg.output_dir, f"preference_data/{date_and_time}")
        if os.path.exists(self.image_save_path):
            shutil.rmtree(self.image_save_path)
        os.makedirs(self.image_save_path, exist_ok=False)
        os.makedirs(self.dataset_save_path, exist_ok=False)

        self.cfg = cfg

    def reinitialize_trainloader(self, cfg: DictConfig) -> Any:
        dataset = instantiate_with_cfg(cfg, split="train")
        should_shuffle = True
        trainloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers,
        )
        return trainloader

    def load_dataloaders(self, cfg: DictConfig, split=None) -> Any:
        """
        If split is not provided, train, validation and test dataloaders are returned in a dict
        If split is provided a single dataloader of specified split is returned
        """
        dataloaders = {}
        if split is None:
            for split in [cfg.train_split_name, cfg.valid_split_name, cfg.test_split_name]:
                dataset = instantiate_with_cfg(cfg, split=split)
                should_shuffle = split == cfg.train_split_name
                dataloaders[split] = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=should_shuffle,
                    batch_size=cfg.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=cfg.num_workers
                )
            return dataloaders
        else:
            dataset = instantiate_with_cfg(cfg, split=split)
            should_shuffle = split == cfg.train_split_name
            return torch.utils.data.DataLoader(
                dataset,
                shuffle=should_shuffle,
                batch_size=cfg.batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=cfg.num_workers,
            )

    def load_optimizer(self, cfg: DictConfig, model: nn.Module):
        optimizer = instantiate(cfg, model=model)
        return optimizer

    def load_scheduler(self, cfg: DictConfig, optimizer):
        scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
        return scheduler

    def load_task(self, cfg: DictConfig, accelerator: BaseAccelerator):
        task = instantiate_with_cfg(cfg, accelerator=accelerator)
        return task

    def verify_or_write_config(self, cfg: TrainerConfig, logger):
        os.makedirs(cfg.output_dir, exist_ok=True)
        yaml_path = os.path.join(cfg.output_dir, "config.yaml")
        OmegaConf.save(cfg, yaml_path, resolve=True) # TODO

        # if not os.path.exists(yaml_path):
        #     OmegaConf.save(cfg, yaml_path, resolve=True) # TODO
        # with open(yaml_path) as f:
        #     existing_config = f.read()
        # if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
        #     raise ValueError(f"Config was not saved correctly - {yaml_path}")
        logger.info(f"Config can be found in {yaml_path}")

    def train(self, image_batch, prompt, epoch):

        #######################################################
        ########### Active Query and Dataset Update ###########
        #######################################################
        queries = self.query_generator.generate_queries(
            images=image_batch,
            query_algorithm="random", # TODO - add to config
            n_queries=10, # TODO - add to config
        )

        # Collect preferences
        self.feedback_interface.reset_dataset() # clear data from previous iteration
        self.feedback_interface.query_batch(
            prompt=prompt,
            image_batch=image_batch,
            query_indices=queries,
        )

        # Save new dataset and reinitialize dataloaders
        self.feedback_interface.save_dataset(dataset_save_path=os.path.join(self.dataset_save_path, f"epoch{epoch}.parquet"))
        self.feedback_interface.save_dataset(dataset_save_path="rl4dgm/my_dataset/my_dataset_train.parquet") 

        # Initialize trainloader from new dataset
        trainloader = self.load_dataloaders(self.cfg.dataset, split="train")
        
        # TODO update validloader 
        validloader = self.load_dataloaders(self.cfg.dataset, split="validation")

        trainloader, validloader = self.accelerator.prepare(trainloader, validloader)
        split2dataloader = {
            "train" : trainloader,
            "validation" : validloader,
        }
        dataloaders = list(split2dataloader.values())

        # trainloader = self.reinitialize_trainloader(self.cfg.dataset)
        # trainloader = self.accelerator.prepare(trainloader)
        # self.split2dataloader["train"] = trainloader
        # self.dataloaders = list(self.split2dataloader.values())

        self.accelerator.recalc_train_length_after_prepare(len(split2dataloader[self.cfg.dataset.train_split_name]))


        #######################################################
        ################ Reward Model Training ################
        #######################################################
        # Train reward model for n epochs

        train_loss, lr = 0.0, 0.0
        for inner_epoch in range(self.accelerator.cfg.num_epochs): # TODO config should have something like reward epochs per loop
            print("Epoch ", inner_epoch)
            for step, batch in enumerate(split2dataloader[self.cfg.dataset.train_split_name]):
                if self.accelerator.should_skip(inner_epoch, step):
                    self.accelerator.update_progbar_step()
                    continue

                if self.accelerator.should_eval():
                    pass
                    # TODO where should eval() be defined? 
                    # validation dataset can be accumulation of all previous human feedbacks?
                    # evaluate()

                if self.accelerator.should_save():
                    self.accelerator.save_checkpoint()

                self.model.train()

                with self.accelerator.accumulate(self.model):
                    loss = self.task.train_step(self.model, self.criterion, batch)
                    avg_loss = self.accelerator.gather(loss).mean().item()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters())

                    breakpoint()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                train_loss += avg_loss / self.accelerator.cfg.gradient_accumulation_steps

                if self.accelerator.sync_gradients:
                    self.accelerator.update_global_step(train_loss)
                    train_loss = 0.0

                if self.accelerator.global_step > 0:
                    try:
                        lr = lr_scheduler.get_last_lr()[0]
                    except:
                        print("get_last_lr exception. Setting lr=0.0")
                        lr = 0.0
                self.accelerator.update_step(avg_loss, lr)

                if self.accelerator.should_end():
                    evaluate()
                    self.accelerator.save_checkpoint()
                    break

            if self.accelerator.should_end():
                break

            self.accelerator.update_epoch()

        self.accelerator.wait_for_everyone()

