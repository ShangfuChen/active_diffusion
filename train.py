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
import torchvision
import numpy as np
import os
import numpy as np
from PIL import Image
import datetime

from rl4dgm.utils.test_pickscore import score_images
from rl4dgm.utils.reward_processor import RewardProcessor
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface

logger = get_logger(__name__)


def eval_samples(accelerator, images):
    """
    Evaluation function to calculate scores for generated image samples
    Args:
        accelerator (accelerate.Accelerator) : pass in for logging results
        images (Tensor) : a batch of images 
    """
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = images[:, 0, :, :]
    score = np.mean(images)
    accelerator.log(
        {"eval_score": score}
    )


###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)

    # create directories to save sampled images
    img_save_dir = os.path.join("/home/shangfu/sampled_images", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    os.mkdir(img_save_dir)
    
    
    ddpo_trainer = DDPOTrainer(config=cfg.ddpo_conf, logger=logger, accelerator=None)#, dummy_loader=dummy_loader)
    reward_processor = RewardProcessor(
        distance_thresh=9.0,
        distance_type="l2",
        reward_error_thresh=2.0,
    )
    feedback_interface = HumanFeedbackInterface(feedback_type="score-one")

    # reward_model_trainer = PickScoreTrainer(cfg=cfg, logger=logger, accelerator=ac)
    # prompt = "a cute cat" # TODO - get from user input?

    for loop in range(20):
        samples, features, prompts, ai_rewards = ddpo_trainer.sample(logger=logger, epoch=loop) #, reward_model=reward_model_trainer.model, processor=reward_model_trainer.processor)
        save_batch_to_images(image_batch=samples, epoch=loop, save_dir=img_save_dir)
        ai_rewards = [elem for sublist in ai_rewards for elem in sublist] # flatten list 
        final_rewards = reward_processor.compute_consensus_rewards(
            images=samples,
            features=features,
            prompts=["a cute cat"]*samples.shape[0],
            ai_rewards=ai_rewards,
            feedback_interface=feedback_interface,
        )
        
        # dummy_rewards = np.ones(samples.shape[0]).tolist()
        ddpo_trainer.train_from_reward_labels(logger=logger, epoch=loop, raw_rewards=final_rewards)

def save_batch_to_images(image_batch, epoch, save_dir):
    save_folder = os.path.join(save_dir, f"epoch{epoch}")
    print(f"saving images to {save_folder}")
    os.mkdir(save_folder)
    for i, im in enumerate(image_batch):
        pil_im = torchvision.transforms.functional.to_pil_image(im)
        pil_im.save(os.path.join(save_folder, f"{i}.jpg"))

if __name__ == "__main__":
    # app.run(main)
    main()
