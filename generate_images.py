
import os 
import numpy as np
import random
import hydra
from ddpo_trainer import DDPOTrainer

import torch 
from PickScore.trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import Accelerator
from accelerate.logging import get_logger


@hydra.main(version_base=None, config_path="/home/hayano/active_diffusion/PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    logger = get_logger(__name__)

    # set seed
    np.random.seed(cfg.ddpo_conf_inference.seed)
    torch.manual_seed(cfg.ddpo_conf_inference.seed)
    random.seed(cfg.ddpo_conf_inference.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.ddpo_conf_inference.seed)
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.ddpo_conf_inference.logdir, cfg.ddpo_conf_inference.run_name),
        automatic_checkpoint_naming=True,
        total_limit=cfg.ddpo_conf_inference.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.ddpo_conf_inference.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train_gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=cfg.ddpo_conf_inference.train_gradient_accumulation_steps*cfg.ddpo_conf_inference.train_num_update)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="debug",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.ddpo_conf_inference.run_name}},
        )
    ddpo_trainer = DDPOTrainer(
        config=cfg.ddpo_conf_inference, 
        logger=logger, 
        # accelerator=accelerator
    )

    # make directory to save the generated images
    img_save_dir = cfg.ddpo_conf_inference.img_save_dir
    os.makedirs(img_save_dir, exist_ok=True)
    high_reward_latents = None

    # if we loaded from a checkpoint, load latents as well
    if len(cfg.ddpo_conf_inference.resume_from) > 0:
        ckpt_epoch = cfg.ddpo_conf_inference.epoch
        if cfg.ddpo_conf_inference.sample_from_best_latent:
            # we only sample from the best latent
            high_reward_latents = torch.load(os.path.join(cfg.ddpo_conf_inference.ckpt_dir, "latents", f"best_{ckpt_epoch}.pt"))
        elif cfg.ddpo_conf_inference.sample_from_all_good_latents:
            # load best and good latents
            best_latent = torch.load(os.path.join(cfg.ddpo_conf_inference.ckpt_dir, "latents", f"best_{ckpt_epoch}.pt"))
            pos_latents = torch.load(os.path.join(cfg.ddpo_conf_inference.ckpt_dir, "latents", f"positives_{ckpt_epoch}.pt"))
            high_reward_latents = torch.cat([pos_latents, best_latent], dim=0)        
        else:
            high_reward_latents = None

    samples, all_latents, prompts, ai_rewards = ddpo_trainer.sample(
        logger=logger,
        epoch=0,
        save_images=True,
        img_save_dir=img_save_dir,
        high_reward_latents=high_reward_latents,
    ) 

if __name__ == "__main__":
    main()