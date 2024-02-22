"""
Pipeline for realtime image generation, feedback, and generative model training.
"""



import json
import os
from typing import Any
import shutil

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

from diffusers import DiffusionPipeline

from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
import rl4dgm.utils.generate_images as image_generator
import rl4dgm.utils.query_generator as query_generator

from rl4dgm.utils.create_dummy_dataset import preference_from_ranked_prompts, preference_from_keyphrases

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
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


def reinitialize_trainloader(cfg: DictConfig) -> Any:
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


def load_optimizer(cfg: DictConfig, model: nn.Module):
    optimizer = instantiate(cfg, model=model)
    return optimizer


def load_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task


def verify_or_write_config(cfg: TrainerConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    yaml_path = os.path.join(cfg.output_dir, "config.yaml")
    # if not os.path.exists(yaml_path):
    #     OmegaConf.save(cfg, yaml_path, resolve=True)
    # with open(yaml_path) as f:
    #     existing_config = f.read()
    # if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
    #     raise ValueError(f"Config was not saved correctly - {yaml_path}")
    logger.info(f"Config can be found in {yaml_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)

    accelerator = instantiate_with_cfg(cfg.accelerator)

    if cfg.debug.activate and accelerator.is_main_process:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

    if accelerator.is_main_process:
        verify_or_write_config(cfg)

    logger.info(f"Loading task")
    task = load_task(cfg.task, accelerator)
    logger.info(f"Loading model")
    model = instantiate_with_cfg(cfg.model)
    logger.info(f"Loading criterion")
    criterion = instantiate_with_cfg(cfg.criterion)
    logger.info(f"Loading optimizer")
    optimizer = load_optimizer(cfg.optimizer, model)
    logger.info(f"Loading lr scheduler")
    lr_scheduler = load_scheduler(cfg.lr_scheduler, optimizer)
    logger.info(f"Loading dataloaders")
    split2dataloader = load_dataloaders(cfg.dataset)

    dataloaders = list(split2dataloader.values())
    model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
    split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))
    
    accelerator.load_state_if_needed()

    accelerator.recalc_train_length_after_prepare(len(split2dataloader[cfg.dataset.train_split_name]))

    accelerator.init_training(cfg)

    def evaluate():
        model.eval()
        end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
        logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
        metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
        accelerator.update_metrics(metrics)
        # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    
    #########################################
    # Initialize diffusion model
    diffusion_model = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to("cuda")
    generator = torch.manual_seed(2023)

    # Create directories to save images and dataset
    image_save_path = os.path.join(cfg.output_dir, "image_data") # TODO get this from config
    dataset_save_path = os.path.join(cfg.output_dir, "preference_data") # TODO get this from config
    # TODO - get rid of below (deletes existing folder!)
    if os.path.exists(image_save_path):
        print("WARNING: removing existing", image_save_path)
        shutil.rmtree(image_save_path)
    if os.path.exists(dataset_save_path):
        print("WARNING: removing existing", dataset_save_path)
        shutil.rmtree(dataset_save_path)

    os.makedirs(image_save_path)
    os.makedirs(dataset_save_path)

    # Initialize user feedback UI
    # feedback_interface = HumanFeedbackInterface()
    feedback_interface = AIFeedbackInterface(preference_function=preference_from_keyphrases)

    # Get user input prompt
    prompt = "A cute cat" # TODO get it as user input
    #########################################


    logger.info(f"task: {task.__class__.__name__}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
    logger.info(
        f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")
    logger.info(f"criterion: {criterion.__class__.__name__}")
    logger.info(f"num. train examples: {len(split2dataloader[cfg.dataset.train_split_name].dataset)}")
    logger.info(f"num. valid examples: {len(split2dataloader[cfg.dataset.valid_split_name].dataset)}")
    logger.info(f"num. test examples: {len(split2dataloader[cfg.dataset.test_split_name].dataset)}")

    for epoch in range(accelerator.cfg.num_epochs):
        print("Epoch ", epoch)

        #########################################

        # Generate new images TODO - currently assumes all images are generated using the same prompt
        # START HERE !!!! TODO - update below parts to take AI feedback
        img_save_dir = os.path.join(image_save_path, f"epoch{epoch}")
        
        # generate cat images
        image_generator.generate_cat_images(
            model=diffusion_model,
            img_save_dir=img_save_dir,
            n_images=40,
            generator=torch.manual_seed(epoch),
            n_inference_steps=10,
        )

        # Generate queries
        # image_directories = []
        # for subdir in os.listdir(img_save_dir):
        #     files = os.listdir(os.path.join(img_save_dir, subdir))
        #     image_directories += [os.path.join(img_save_dir, subdir, file) for file in files]
        image_directories = [os.path.join(img_save_dir, subdir) for subdir in os.listdir(img_save_dir)]
        print("================= image_directories =================\n", image_directories)
        queries = query_generator.generate_queries(
            image_directories=image_directories,
            query_algorithm="random",
            n_queries=10,
        )

        # Query AI and save new dataset
        feedback_interface.reset_dataset() # get rid of previous epoch data # TODO design choice?
        for query in queries:
            feedback_interface.query(
                keyphrases=["black", "cute"],
                img_paths=query,
                prompt=prompt
            )
        feedback_interface.save_dataset(dataset_save_path=os.path.join(dataset_save_path, f"epoch{epoch}.parquet"))
        
        # image_generator.generate_images(
        #     model=diffusion_model,
        #     img_save_dir=img_save_dir,
        #     prompt=prompt,
        #     n_images=50, # TODO
        #     generator=torch.manual_seed(epoch), # TODO
        #     n_inference_steps=10, # TODO
        # )

        # # Generate queries
        # queries = query_generator.generate_queries(
        #     image_directories=[
        #         img_save_dir,
        #     ],
        #     query_algorithm="random", # TODO get this from config
        #     n_queries=10, # TODO get this from config
        # )

        # # Query human and save new dataset
        # feedback_interface.reset_dataset() # get rid of previous epoch data # TODO design choice?
        # for query in queries:
        #     feedback_interface.query(img_paths=query, prompt=prompt)
        # feedback_interface.save_dataset(dataset_save_path=os.path.join(dataset_save_path, f"epoch{epoch}.parquet"))
        # print("Saved new dataset to ", os.path.join(dataset_save_path, f"epoch{epoch}"))

        # TODO - Below is temporary hack. Fix it to make dataset location point to the newly saved dataset
        # NOTE - dataloader.dataset.cfg contains dataset_loc
        feedback_interface.save_dataset(dataset_save_path="/home/hayano/rl4dgm/rl4dgm/my_dataset/my_dataset_train.parquet")
        print("Overwrote dataset at /home/hayano/rl4dgm/rl4dgm/my_dataset/my_dataset_train.parquet")        
        
        # Re-initialize dataloaders from newly collected dataset
        trainloader = reinitialize_trainloader(cfg.dataset)
        trainloader = accelerator.prepare(trainloader)
        split2dataloader["train"] = trainloader
        dataloaders = split2dataloader.values()

        # TODO TEST
        #########################################
        train_loss, lr = 0.0, 0.0
        for step, batch in enumerate(split2dataloader[cfg.dataset.train_split_name]):
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
    logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
    metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
    accelerator.update_metrics(metrics)
    logger.info(f"*** Evaluating {cfg.dataset.test_split_name} ***")
    metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.test_split_name])
    metrics = {f"{cfg.dataset.test_split_name}_{k}": v for k, v in metrics.items()}
    accelerator.update_metrics(metrics)
    accelerator.unwrap_and_save(model)
    accelerator.end_training()


if __name__ == '__main__':
    main()
