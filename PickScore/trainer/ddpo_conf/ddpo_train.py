from dataclasses import dataclass
from typing import Optional

import torch
import ml_collections


@dataclass
class DDPOTrainConfig:
    # config = ml_collections.ConfigDict()
    n_outer_loops: int = 9 # number of times ddpo train should be called
    save_dataset: bool = True
    dataset_save_path: str = "/home/shangfu/active_diffusion/rl4dgm/realhuman_tests/dataset.parquet"
    project_name: str = "debug"
    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.

    # run_name: str = "query_everything_raw_reward"
    # run_name: str = "ddim_1_hand"
    # run_name: str = "debug"
    run_name: str = "mountain_realhuman_seed0"
    # run_name: str = "narcissus_from_best_noise_0.01"
    # run_name: str = "query_everything_raw_reward_softmax"
    # run_name: str = "query_everything_similarity_to_all_pos"
    # run_name: str = "query_everything_with_similarity"

    # sample_from_best_latent: bool = False
    sample_from_best_latent: bool = True
    reward_mode: str = "similarity-to-best-sample"
    # reward_mode: str = "similarity-to-all-positive"

    save_dir: str = "/data/shangfu"

    # random seed for reproducibility.
    seed: int = 0
    # top-level logging directory for checkpoint saving.
    logdir: str = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    num_epochs: int = 100
    # number of epochs between saving model checkpoints. deactivate by setting a negative value
    save_freq: int = -1
    # number of checkpoints to keep before overwriting old ones.
    num_checkpoint_limit: int = 5
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    mixed_precision: str = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    allow_tf32: bool = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    # resume_from: str = "logs/ddim_1_hand/checkpoints/checkpoint_0"
    resume_from: str = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    use_lora: bool = True

    ###### Pretrained Model ######
    pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    # pretrained_model: str = "stabilityai/stable-diffusion-2-1"
    # revision of the model to load.
    pretrained_revision: str = "main"

    ###### Sampling ######
    # config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample_num_steps: int = 50
    # sample_num_steps: int = 5
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample_eta: float = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample_guidance_scale: float = 5.0
    # batch size (per GPU!) to use for sampling.
    sample_batch_size: int = 4
    # sample_batch_size: int = 2
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample_num_batches_per_epoch: int = 16
    # sample_num_batches_per_epoch: int = 25
    # sample_num_batches_per_epoch: int = 32
    
    ###### Training ######
    # config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train_batch_size: int = 2
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train_use_8bit_adam: bool = False
    # learning rate.
    train_learning_rate: float = 3e-4
    # Adam beta1.
    train_adam_beta1: float = 0.9
    # Adam beta2.
    train_adam_beta2: float = 0.999
    # Adam weight decay.
    train_adam_weight_decay: float = 1e-4
    # Adam epsilon.
    train_adam_epsilon: float = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`
    train_gradient_accumulation_steps: int = 4
    # maximum gradient norm for gradient clipping.
    train_max_grad_norm: float = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train_num_inner_epochs: int = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train_cfg: bool = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train_adv_clip_max: int = 5
    # the PPO clip range.
    train_clip_range: float = 1e-4
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train_timestep_fraction: float = 1.0
    train_num_update: int = 5
    normalization: bool = True
    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    # prompt_fn: str = "simple_animals"
    # prompt_fn: str = "cute_cats"
    # prompt_fn: str = "ugly_cats"
    prompt_fn: str = "test_prompt"
    # prompt_fn: str = "cute_animals"
    # kwargs to pass to the prompt function.

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    # checkpoint path for pickscore model
    # reward_fn: str = "jpeg_compressibility"
    # reward_fn: str = "color_score"
    # reward_fn: str = "color_score_09_discrete"
    reward_fn: str = "aesthetic_score"
    # reward_fn: str = "pickscore"
    ckpt_path: str = "ddpo/trained_reward_model/checkpoint-final"

    ###### Per-Prompt Stat Tracking ######
    # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
    # advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
    # advantages will be calculated using the mean and std of the entire batch.
    # config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    per_prompt_stat_tracking: Optional[bool] = None
    # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    per_prompt_stat_tracking_buffer_size: int = 16
    # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
    # contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    per_prompt_stat_tracking_min_count: int = 16
