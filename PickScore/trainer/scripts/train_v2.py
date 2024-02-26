###### DDPO ######
from collections import defaultdict
import contextlib
import sys
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo.prompts
import ddpo.rewards
from ddpo.stat_tracking import PerPromptStatTracker
from ddpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
import random
from functools import partial
import tqdm
import tempfile
from PIL import Image
from transformers import AutoProcessor, AutoModel
from PickScore.trainer.scripts.mystep_realtime import reward_model_setup, reward_train_step

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
# Add PickScore directory
# sys.path.append('PickScore')

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)
###### DDPO ######

###### PickScore imports ######
import json
import os
from typing import Any
import shutil

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
from rl4dgm.utils.generate_images import ImageGenerator
from rl4dgm.utils.query_generator import QueryGenerator
from rl4dgm.utils.create_dummy_dataset import preference_from_ranked_prompts, preference_from_keyphrases

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
###### PickScore imports ######


###### PickScore helpers ######
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

###### PickScore helpers ######

###### Main training loop ######
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

    image_generator = ImageGenerator()
    query_generator = QueryGenerator()

    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)

    # def evaluate():
    #     model.eval()
    #     end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
    #     logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
    #     metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
    #     accelerator.update_metrics(metrics)
    #     # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    
    #########################################
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
    feedback_interface = HumanFeedbackInterface()
    # feedback_interface = AIFeedbackInterface(preference_function=preference_from_keyphrases)

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


    ################ DDPO ################
    ddpo_cfg = cfg.ddpo_conf
    train_ddpo(config=ddpo_cfg)

    ################ TODO call PickScore training step ################
    


def train_ddpo(config):
    # basic Accelerate and logging setup
    # config = all_configs['ddpo_conf']
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample_num_steps * config.train_timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train_gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=\
          config.train_gradient_accumulation_steps*config.train_num_update)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="active-diffusion",
            # config=config.to_dict(),
            config=dict(config),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model, revision=config.pretrained_revision
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train_use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train_learning_rate,
        betas=(config.train_adam_beta1, config.train_adam_beta2),
        weight_decay=config.train_adam_weight_decay,
        eps=config.train_adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo.rewards, config.reward_fn)()

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train_batch_size, 1, 1)

    # initialize stat tracker - TODO missing entry in config
    # if config.per_prompt_stat_tracking:
    if False:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking_buffer_size,
            config.per_prompt_stat_tracking_min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    # TODO - below line throws error about double-defining optimizer in code and config
    # using second line throws error about CUDA_HOME environment variable not being set
    unet, optimizer = accelerator.prepare(unet, optimizer) 
    # unet = accelerator.prepare(unet) 

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=4)

    # Train!
    samples_per_epoch = (
        config.sample_batch_size
        * accelerator.num_processes
        * config.sample_num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train_batch_size
        * accelerator.num_processes
        * config.train_gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample_batch_size}")
    logger.info(f"  Train batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train_gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train_num_inner_epochs}")

    assert config.sample_batch_size >= config.train_batch_size
    assert config.sample_batch_size % config.train_batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1 # TODO this entry missing from config?
    else:
        first_epoch = 0

    ###### TODO get it reward model as input to this function?  #####
    # Load Pickscore model and preprocessor
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    # pickscore_model = AutoModel.from_pretrained(config.ckpt_path).eval().to(accelerator.device)
    pickscore_model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").eval().to(accelerator.device)

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[
                    prompt_fn()
                    for _ in range(config.sample_batch_size)
                ]
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                images, _, latents, log_probs = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample_num_steps,
                    guidance_scale=config.sample_guidance_scale,
                    eta=config.sample_eta,
                    output_type="pt",
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            # other reward functions besides the pickscore
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # TODO pass. Calculate rewards after pickscore model finetuning
            rewards = executor.submit(reward_fn,
                                      processor,
                                      pickscore_model,
                                      images,
                                      prompts,
                                      device=accelerator.device)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "images": images,
                }
            )

        ################## Update RM and Calculate Rewards ##################
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        breakpoint() 
        # TODO - call train_reward_model here


        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                # pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample_batch_size * config.sample_num_batches_per_epoch
        )
        assert num_timesteps == config.sample_num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                
                for k in tqdm(
                    range(config.train_num_update),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    j = random.randint(0, num_timesteps-1)
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train_cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample_guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            # sample["latent"][:, j] (B, 4, 64, 64)
                            # sample["timesteps"][:, j] (B)
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample_eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -config.train_adv_clip_max,
                            config.train_adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        info["ratio"].append(ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train_clip_range,
                            1.0 + config.train_clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train_clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train_max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == num_train_timesteps - 1) and (
                            # i + 1
                        # ) % config.train_gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

def train_reward_model(images, reward_model, processor, dataset_save_path):
    """
    Takes a reward model training step: generate queries, collect feedback, train reward model,
    and return rewards from the newly-trained reward model.

    Args:
        images (Tensor) : batch of images generated by DGM (B x C x H x W)
        reward_model (AutoModel) : PickScore model to be used as reward model

    Returns:
        rewards (Tensor) : Tensor of rewards for each of the input images
        reward_model (AutoModel) : finetuned reward model
    """
    #######################################################
    ########### Active Query and Dataset Update ###########
    #######################################################

    # Generate queries
    queries = query_generator.generate_queries_from_tensor(
        image_batch=images,
        query_algorithm="random", # TODO - add to config?
        n_queries=10, # TODO - add to config?
    )

    # Collect preferences
    feedback_interface.reset_dataset() # clear data from previous iteration
    feedback_interface.query_batch(
        prompt=prompt,
        image_batch=images,
        query_indices=queries,
    )

    # Save new dataset and reinitialize dataloaders
    feedback_interface.save_dataset(dataset_save_path=os.path.join(dataset_save_path, f"epoch{epoch}.parquet"))
    feedback_interface.save_dataset(dataset_save_path="../rl4dgm/my_dataset/my_dataset_train.parquet")
    trainloader = reinitialize_trainloader(cfg.dataset)
    trainloader = accelerator.prepare(trainloader)
    split2dataloader["train"] = trainloader
    dataloaders = split2dataloader.values()

    #######################################################
    ################ Reward Model Training ################
    #######################################################
    # Train reward model for n epochs

    train_loss, lr = 0.0, 0.0
    for epoch in range(accelerator.cfg.num_epochs): # TODO config should have something like reward epochs per loop
        print("Epoch ", epoch)
        for step, batch in enumerate(split2dataloader[cfg.dataset.train_split_name]):
            if accelerator.should_skip(epoch, step):
                accelerator.update_progbar_step()
                continue

            if accelerator.should_eval():
                evaluate()

            if accelerator.should_save():
                accelerator.save_checkpoint()

            reward_model.train()

            with accelerator.accumulate(reward_model):
                loss = task.train_step(reward_model, criterion, batch)
                avg_loss = accelerator.gather(loss).mean().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(reward_model.parameters())

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

    #######################################################
    ################### Compute Rewards ###################
    #######################################################
    # Compute reward for input images
    probs, scores = score_images(prompt=prompt, reward_model=reward_model, processor=processor, images=images)

    return scores, reward_model




if __name__ == "__main__":
    # app.run(main)
    main()
