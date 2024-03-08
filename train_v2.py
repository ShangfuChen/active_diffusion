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
from reward_model_trainer import PickScoreTrainer
# from reward_model_trainer_copy import PickScoreTrainer

from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoModel
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg
import tqdm
import torch
import os
import numpy as np

from rl4dgm.utils.test_pickscore import score_images

logger = get_logger(__name__)

###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)
    # breakpoint()
    # dummy reward model
    # processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # processor = AutoProcessor.from_pretrained(processor_name_or_path)
    # reward_model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").to("cuda").eval()
    # reward_accelerator = instantiate_with_cfg(cfg.accelerator)
    # print("\nACCELERATE_USE_DEEPSPEED: ", os.environ.get("ACCELERATE_USE_DEEPSPEED"))
    # print("\nReward accelerator: ", reward_accelerator)
    # breakpoint()
    # breakpoint()
    # dummy_loader = reward_model_trainer.load_dataloaders(reward_model_trainer.cfg.dataset, split="train")
    # breakpoint()
    # from accelerate import Accelerator
    # ac = Accelerator(
    #     gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
    #     mixed_precision=cfg.accelerator.mixed_precision,
    #     log_with=cfg.accelerator.log_with,
    #     project_dir=cfg.accelerator.output_dir,
    #     dynamo_backend=cfg.accelerator.dynamo_backend,
    # )
    # breakpoint()

    from PickScore.trainer.configs.configs import TrainerConfig, instantiate_with_cfg
    ac = instantiate_with_cfg(cfg.accelerator)

    ddpo_trainer = DDPOTrainer(config=cfg.ddpo_conf, logger=logger, accelerator=ac.accelerator)#, dummy_loader=dummy_loader)
    reward_model_trainer = PickScoreTrainer(cfg=cfg, logger=logger, accelerator=ac)
    # prompt = "a cute cat" # TODO - get from user input?

    _, prev_scores = score_images(
        model=reward_model_trainer.model.module, 
        processor=reward_model_trainer.processor, 
        prompt=prompt, 
        img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
    )
    for loop in range(5):
        print("Loop epoch ", loop)
        
        probs, scores = score_images(
            model=reward_model_trainer.model.module, 
            processor=reward_model_trainer.processor, 
            prompt=prompt, 
            img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
        )
        # breakpoint()

        samples, prompts = ddpo_trainer.sample(logger=logger, epoch=loop, reward_model=reward_model_trainer.model, processor=reward_model_trainer.processor)
        # breakpoint()
        # dummy samples for PickScore testing
        # samples = torch.Tensor()
        # samples = samples.new_zeros(size=[32,3,256,256])

        reward_model_trainer.train(image_batch=samples, epoch=loop, prompts=prompts, logger=logger)
        _, scores = score_images(
            model=reward_model_trainer.model.module, 
            processor=reward_model_trainer.processor, 
            prompt=prompt, 
            img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
        )
        print("\nMean Delta Scores: ", np.mean(np.array(scores) - np.array(prev_scores)))
        ac.log({
            "mean_delta_scores" : np.mean(np.array(scores) - np.array(prev_scores)),
        })
        prev_scores = scores
        probs, scores = score_images(
            model=reward_model_trainer.model.module, 
            processor=reward_model_trainer.processor, 
            prompt=prompt, 
            img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
        )
        ddpo_trainer.train(logger=logger, epoch=loop, reward_model=reward_model_trainer.model, processor=reward_model_trainer.processor)

    # iteration = 0
    # samples, prompts = ddpo_trainer.sample(logger=logger, epoch=iteration, reward_model=reward_model_trainer.model, processor=reward_model_trainer.processor)
    # _, prev_scores = score_images(
    #     model=reward_model_trainer.model.module, 
    #     processor=reward_model_trainer.processor, 
    #     prompt=prompt, 
    #     img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
    # )
    # while True:
    #     print("\niteration", iteration)
    #     reward_model_trainer.train(image_batch=samples, epoch=iteration, prompts=prompts, logger=logger)
    #     _, scores = score_images(
    #         model=reward_model_trainer.model.module, 
    #         processor=reward_model_trainer.processor, 
    #         prompt=prompt, 
    #         img_dir="/home/hayano/active_diffusion/poc_datasets/ordered_cat_bw/image_data"
    #     )
    #     print("\nDelta Scores: ", np.array(scores) - np.array(prev_scores))
    #     prev_scores = scores

if __name__ == "__main__":
    # app.run(main)
    main()
