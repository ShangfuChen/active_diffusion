"""
Main training script that calls DGM and reward model training in alternating fashion

- imports
- from hydra config
    - initialize_dgm_training(cfg.ddpo) --> model, optimizer, dataloaders, accelerator
    - initialize_reward_training(cfg.reward) --> model, optimizer, dataloaders, accelerator
    - initialize logger
- train DGM --> train reward model --> train DGM --> ...

ddpo.sample()

for _ in range(ddpo_training_epochs_per_loop):
    ddpo.train(dgm, reward_model, dgm_optimizer, dgm_accelerator, dgm_dataset?) --> None

for _ in range(reward_model_training_epochs_per_loop):
    reward_model.train(reward_model, samples, reward_optimizer, reward_accelerator, reward_dataset?) --> rewards, reward_model 

"""
import hydra

from ddpo_trainer import DDPOTrainer

from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoModel
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg
import tqdm

logger = get_logger(__name__)

###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)

    # dummy reward model
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    reward_model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").to("cuda").eval()
    # reward_accelerator = instantiate_with_cfg(cfg.accelerator)

    ddpo_trainer = DDPOTrainer(config=cfg.ddpo_conf, logger=logger)

    samples = ddpo_trainer.sample(logger=logger, reward_model=reward_model, processor=processor)
    global_step = ddpo_trainer.train(logger=logger, epoch=0)

if __name__ == "__main__":
    # app.run(main)
    main()
