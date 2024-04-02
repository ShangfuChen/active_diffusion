
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
from rl4dgm.user_feedback_interface.preference_functions import ColorPickOne
from rl4dgm.utils.query_generator import QueryGenerator
from rl4dgm.utils.create_dummy_dataset import preference_from_ranked_prompts, preference_from_keyphrases

from rl4dgm.utils.generate_images import ImageGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from accelerate import Accelerator
from trainer.accelerators.utils import get_nvidia_smi_gpu_memory_stats_str, print_config, _flatten_dict
import math

class PickScoreTrainer:
    def __init__(self, cfg : DictConfig, logger, accelerator=None):
        
        """
        Args:
            cfg (DicstConfig) : config for PickScore training
            logger (accelerate logging.logger) : logger to write logs to 
            accelerator (accelerate.Accelerator) : If provided, use the provided accelerator. If not, initialize from config
        """

        self.accelerator = instantiate_with_cfg(cfg.accelerator) if accelerator is None else accelerator

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
        
        validloader = self.load_dataloaders(cfg.dataset, split="validation")
        
        self.model, self.optimizer, self.lr_scheduler, validloader= self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, validloader)

        self.accelerator.load_state_if_needed()
        self.accelerator.init_training(cfg)

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)

        # initialize feedback interface 
        if cfg.query_conf.feedback_agent == "human":
            self.feedback_interface = HumanFeedbackInterface()
        elif cfg.query_conf.feedback_agent == "ai":
            self.feedback_interface = AIFeedbackInterface(preference_function=ColorPickOne)
        else:
            raise Exception(f"human is the only feedback agent currently supported. Got {cfg.query_conf.feedback_agent}")

        # initialize query generator
        self.query_generator = QueryGenerator()

        # get query algorithm type from config
        self.query_method = cfg.query_conf.query_algorithm
        assert self.query_method in self.query_generator.QUERY_ALGORITHMS.keys(), f"query should be one of {self.query_generator.QUERY_ALGORITHMS.keys()}. Got {self.query_method}"

        # Create files to save images and dataset to
        if self.accelerator.is_main_process:
            self.image_save_path = os.path.join(cfg.output_dir, "image_data")
            date_and_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.dataset_save_path = os.path.join(cfg.output_dir, f"preference_data/{date_and_time}")
            if os.path.exists(self.image_save_path):
                shutil.rmtree(self.image_save_path)
            os.makedirs(self.image_save_path, exist_ok=False)
            os.makedirs(self.dataset_save_path, exist_ok=False)

        self.split2dataloader = None
        self.cfg = cfg

    def evaluate(self, logger):
        self.model.eval()
        end_of_train_dataloader = self.accelerator.gradient_state.end_of_dataloader
        logger.info(f"*** Evaluating {self.cfg.dataset.valid_split_name} ***")
        metrics = self.task.evaluate(self.model, self.criterion, self.split2dataloader[self.cfg.dataset.valid_split_name])
        self.accelerator.update_metrics(metrics)
        # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    def color_evaluate(self, prefix):
        # Calculate color score and log on wandb
        self.model.eval()
        # image_inputs = self.processor(
            # images=images,
            # padding=True,
            # truncation=True,
            # max_length=77,
            # return_tensors="pt",
        # ).to(self.accelerator.device)
        
        correct = 0
        total = 0
        for batch in self.split2dataloader[prefix]:
            im0 = batch["pixel_values_0"]
            im1 = batch["pixel_values_1"]
            # label0 = batch["label_0"]
            # label1 = batch["label_1"]
            B = im0.shape[0]
            image_inputs = {'pixel_values': torch.cat((im0, im1), dim=0)}
            # NOTE: a HACK for one prompt
            prompts = ('a cute cat')*(2*B)
            # NOTE: processor is a tokenizer that 
            # Image: convert (3, 512, 512) images to (3, 224, 224) pixel values 
            # Text: convert str to features that CLIP encoder understands
            text_inputs = self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.accelerator.device)
            with torch.no_grad():
                # embed
                try:
                    image_embs = self.model.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                    text_embs = self.model.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                
                    # score
                    scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                except:
                    image_embs = self.model.module.get_image_features(**image_inputs)
                    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                    text_embs = self.model.module.get_text_features(**text_inputs)
                    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                    # score
                    scores = self.model.module.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            score0, score1 = torch.split(scores.cpu(), [B, B])
            predictions = (score0 > score1)
            # labels = (label0 > label1).cpu() 

            # images = (images * 255).round().clamp(0, 255).cpu()
            red_score0 = torch.mean(im0[:, 0, :, :], dim=(-2, -1))
            red_score1 = torch.mean(im1[:, 0, :, :], dim=(-2, -1))
            # red_scores = red_scores.reshape(2, -1)
            labels = (red_score0 > red_score1).cpu()
            correct += (predictions == labels).sum()
            total += len(predictions)
        accuracy = correct/total
        self.accelerator.log({f"reward model {prefix} accuracy: ": accuracy})
        


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
        return trainloadepreference_functionr

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
        if cfg._target_ == "torch.optim.adamw.AdamW":
            optimizer = instantiate(config=cfg, params=model.parameters())
        else:
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
        #     OmegaConf.save(cfg, yaml_path, resolve=True)
        # with open(yaml_path) as f:
        #     existing_config = f.read()
        # if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
        #     raise ValueError(f"Config was not saved correctly - {yaml_path}")
        logger.info(f"Config can be found in {yaml_path}")

    def train(self, image_batch, prompts, epoch, logger):

        # TODO - logging

        #######################################################
        ########### Active Query and Dataset Update ###########
        #######################################################
        if self.accelerator.is_main_process:
            queries, query_prompts = self.query_generator.generate_queries(
                images=image_batch,
                query_algorithm=self.query_method, # TODO - add to config
                n_queries=self.cfg.query_conf.n_feedbacks_per_query, # TODO - add to config
                prompts=prompts,
            )

            # Collect preferences
            self.feedback_interface.reset_dataset() # clear data from previous iteration
            self.feedback_interface.query_batch(
                prompts=query_prompts,
                image_batch=image_batch,
                query_indices=queries,
            )

            # Save new dataset and reinitialize dataloaders - TODO
            self.feedback_interface.save_dataset(dataset_save_path=os.path.join(self.dataset_save_path, f"epoch{epoch}.parquet"))
            self.feedback_interface.save_dataset(dataset_save_path="rl4dgm/my_dataset/my_dataset_train.parquet") 

        # wait for the main process to update the training dataset
        self.accelerator.wait_for_everyone()

        # Initialize trainloader from new dataset
        trainloader = self.load_dataloaders(self.cfg.dataset, split="train")
        
        # TODO update validloader 
        validloader = self.load_dataloaders(self.cfg.dataset, split="validation")

        trainloader, validloader = self.accelerator.prepare(trainloader, validloader)
        self.split2dataloader = {
            "train" : trainloader,
            "validation" : validloader,
        }
        dataloaders = list(self.split2dataloader.values())
        #######################################################
        ################ Reward Model Training ################
        #######################################################
        # Train reward model for n epochs
        train_loss, lr = 0.0, 0.0
        for inner_epoch in range(self.cfg.accelerator.num_epochs): # TODO config should have something like reward epochs per loop
            for step, batch in enumerate(self.split2dataloader[self.cfg.dataset.train_split_name]):
                if self.accelerator.should_eval():
                    # TODO - validation dataset can be accumulation of all previous human feedbacks?
                    pass
                    # self.evaluate(logger=logger)
                # TODO #
                # Skip model saving now
                # if self.accelerator.should_save():
                    # self.accelerator.save_checkpoint()

                self.model.train()

                with self.accelerator.accumulate(self.model):
                    loss = self.task.train_step(self.model, self.criterion, batch)
                    avg_loss = self.accelerator.gather(loss).mean().item()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters())

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                train_loss += avg_loss / self.cfg.accelerator.gradient_accumulation_steps
                if self.accelerator.sync_gradients:
                    self.accelerator.update_global_step(train_loss)
                    train_loss = 0.0

                if self.accelerator.global_step > 0:
                    try:
                        lr = self.lr_scheduler.get_last_lr()[0]
                    except:
                        print("get_last_lr exception. Setting lr=0.0")
                        lr = 0.0

                self.accelerator.lr = lr
                # if self.accelerator.should_end():
                    # self.evaluate(logger=logger)
                    # self.accelerator.save_checkpoint()
                    # break
                self.accelerator.step += 1

            # if self.accelerator.should_end():
                # break

            self.accelerator.update_epoch()

        self.accelerator.wait_for_everyone()


class PickScoreStaticDatasetTrainer(PickScoreTrainer):
    def __init__(self, cfg : DictConfig):
        self.cfg = cfg
        self.logger = get_logger(__name__)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train(self,):
        accelerator = instantiate_with_cfg(self.cfg.accelerator)
        if self.cfg.debug.activate and accelerator.is_main_process:
            import pydevd_pycharm
            pydevd_pycharm.settrace('localhost', port=self.cfg.debug.port, stdoutToServer=True, stderrToServer=True)

        if accelerator.is_main_process:
            self.verify_or_write_config(self.cfg, self.logger)

        self.logger.info(f"Loading task")
        task = self.load_task(self.cfg.task, accelerator)
        self.logger.info(f"Loading model")
        model = instantiate_with_cfg(self.cfg.model)
        self.logger.info(f"Loading criterion")
        criterion = instantiate_with_cfg(self.cfg.criterion)
        self.logger.info(f"Loading optimizer")
        optimizer = self.load_optimizer(self.cfg.optimizer, model)
        self.logger.info(f"Loading lr scheduler")
        lr_scheduler = self.load_scheduler(self.cfg.lr_scheduler, optimizer)
        self.logger.info(f"Loading dataloaders")
        split2dataloader = self.load_dataloaders(self.cfg.dataset)

        dataloaders = list(split2dataloader.values())
        model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
        split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))

        accelerator.load_state_if_needed()

        accelerator.recalc_train_length_after_prepare(len(split2dataloader[self.cfg.dataset.train_split_name]))

        accelerator.init_training(self.cfg)

        def evaluate():
            model.eval()
            end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
            self.logger.info(f"*** Evaluating {self.cfg.dataset.valid_split_name} ***")
            metrics = task.evaluate(model, criterion, split2dataloader[self.cfg.dataset.valid_split_name])
            accelerator.update_metrics(metrics)
            # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

        self.logger.info(f"task: {task.__class__.__name__}")
        self.logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
        self.logger.info(f"model: {model.__class__.__name__}")
        self.logger.info(
            f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")
        self.logger.info(f"criterion: {criterion.__class__.__name__}")
        self.logger.info(f"num. train examples: {len(split2dataloader[self.cfg.dataset.train_split_name].dataset)}")
        self.logger.info(f"num. valid examples: {len(split2dataloader[self.cfg.dataset.valid_split_name].dataset)}")
        self.logger.info(f"num. test examples: {len(split2dataloader[self.cfg.dataset.test_split_name].dataset)}")

        for epoch in range(accelerator.cfg.num_epochs):
            print("Epoch ", epoch)
            train_loss, lr = 0.0, 0.0
            for step, batch in enumerate(split2dataloader[self.cfg.dataset.train_split_name]):
                if accelerator.should_skip(epoch, step):
                    accelerator.update_progbar_step()
                    continue

                if accelerator.should_eval():
                    evaluate()

                if accelerator.should_save():
                    accelerator.save_checkpoint()

                model.train()

                with accelerator.accumulate(model):
                    loss = task.train_step(model, criterion, batch)
                    avg_loss = accelerator.gather(loss).mean().item()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters())

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                train_loss += avg_loss / accelerator.cfg.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    accelerator.update_global_step(train_loss)
                    train_loss = 0.0

                if accelerator.global_step > 0:
                    try:
                        lr = lr_scheduler.get_last_lr()[0]
                    except:
                        print("get_last_lr exception. Setting lr=0.0")
                        lr = 0.0
                accelerator.update_step(avg_loss, lr)

                if accelerator.should_end():
                    evaluate()
                    accelerator.save_checkpoint()
                    break

            if accelerator.should_end():
                break

            accelerator.update_epoch()

        accelerator.wait_for_everyone()
        accelerator.load_best_checkpoint()
        self.logger.info(f"*** Evaluating {self.cfg.dataset.valid_split_name} ***")
        metrics = task.evaluate(model, criterion, split2dataloader[self.cfg.dataset.valid_split_name])
        accelerator.update_metrics(metrics)
        self.logger.info(f"*** Evaluating {self.cfg.dataset.test_split_name} ***")
        metrics = task.evaluate(model, criterion, split2dataloader[self.cfg.dataset.test_split_name])
        metrics = {f"{self.cfg.dataset.test_split_name}_{k}": v for k, v in metrics.items()}
        accelerator.update_metrics(metrics)
        accelerator.unwrap_and_save(model)
        accelerator.end_training()
