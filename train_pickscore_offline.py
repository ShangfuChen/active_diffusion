
import argparse
import hydra

from trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from reward_model_trainer import PickScoreStaticDatasetTrainer

@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    cfg.dataset.dataset_loc = "/home/hayano/active_diffusion/poc_datasets/ordered_cat_black_white/dummy"
    trainer = PickScoreStaticDatasetTrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    
    main()