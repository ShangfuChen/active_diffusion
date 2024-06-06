

import hydra

from accelerate import Accelerator
from accelerate.logging import get_logger
from PickScore.trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from accelerate.utils import set_seed, ProjectConfiguration


import os
import numpy as np
import torch
import wandb
import PIL
import random
import datetime
import tempfile

from ddpo_trainer import DDPOTrainer
from rl4dgm.user_feedback_interface.preference_functions import ColorPickOne, ColorScoreOne, PickScore
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import EvaluationQueryGenerator
from rl4dgm.models.mydatasets import TripletDatasetWithBestSample, TripletDataset, DoubleTripletDataset, FeatureDoubleLabelDataset, HumanDataset
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer, DoubleTripletEncoderTrainer
from rl4dgm.reward_predictor_trainers.reward_predictor_trainers import ErrorPredictorEnsembleTrainerVoting

logger = get_logger(__name__)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)

    # create directories to save sampled images
    img_save_dir = os.path.join("/home/shangfu/sampled_images", cfg.ddpo_conf.run_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=False)
    
    
    n_human_data_for_training = cfg.human_encoder_conf.n_data_needed_for_training # minimum number of human data to accumulate before training the human encoder
    
    # set seed
    np.random.seed(cfg.ddpo_conf.seed)
    torch.manual_seed(cfg.ddpo_conf.seed)
    random.seed(cfg.ddpo_conf.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.ddpo_conf.seed)

    ############################################
    # Initialize accelerator
    ############################################
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.ddpo_conf.logdir, cfg.ddpo_conf.run_name),
        automatic_checkpoint_naming=True,
        total_limit=cfg.ddpo_conf.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.ddpo_conf.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train_gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=cfg.ddpo_conf.train_gradient_accumulation_steps*cfg.ddpo_conf.train_num_update)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="active-diffusion",
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.ddpo_conf.run_name}},
        )


    ############################################
    # Initialize trainers
    ############################################
    torch.manual_seed(cfg.ddpo_conf.seed) # DONT REMOVE THIS
    
    print("\nInitializing DDPO trainer...")
    ddpo_trainer = DDPOTrainer(
        config=cfg.ddpo_conf, 
        logger=logger, 
        accelerator=accelerator
    )
    print("...done\n")

    ############################################
    # Initialize other components
    ############################################
    query_generator = EvaluationQueryGenerator(seed=cfg.ddpo_conf.seed)

    ############################################
    # Initialize feedback interfaces
    ############################################
    if cfg.query_conf.feedback_agent == "human":
        feedback_interface = HumanFeedbackInterface(feedback_type="score-one")
    elif cfg.query_conf.feedback_agent == "ai":
        feedback_interface = AIFeedbackInterface(
            feedback_type="score-one",
            preference_function=PickScore,
        )
    # setup query config
    # random query is the only supported query method now
    if cfg.query_conf.query_type == "random":
        query_kwargs = {"n_queries" : cfg.query_conf.n_feedback_per_query}
    
    ############################################################
    # Initialize human dataset to accumulate
    ############################################################
    human_dataset = HumanDataset(
        n_data_to_accumulate=n_human_data_for_training,
        device=accelerator.device,
    )
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    n_total_human_feedback = 0
    best_sample_latent = None
    best_noise_latent = None
    best_reward = 0

    n_ddpo_train_calls = 0 # number of times ddpo training loop was called
    loop = 0 # number of times the main training loop has run
    
    while n_ddpo_train_calls < cfg.ddpo_conf.n_outer_loops:
        ############################################################
        # Sample from SD model
        ############################################################
        if cfg.ddpo_conf.sample_from_best_latent:
            print("Generate from the best sample latent.")
            samples, all_latents, _, _ = ddpo_trainer.sample(
                logger=logger,
                epoch=loop,
                save_images=True,
                img_save_dir=img_save_dir,
                high_reward_latents=best_noise_latent,
            )
        else:
            print("Generate from random latent.")
            samples, all_latents, _, _ = ddpo_trainer.sample(
                logger=logger,
                epoch=loop,
                save_images=True,
                img_save_dir=img_save_dir,
                high_reward_latents=None,
            )

        sd_features = torch.flatten(all_latents[:,-1,:,:,:], start_dim=1).float()
        sd_noises = all_latents[:, 0,:,:,:].float()
        print("Sampled from SD model and flattened featuers to ", sd_features.shape)

        # ############################################################
        # # Active query --> get human rewards
        # ############################################################
        print("Begin active query")
        if loop == 0:
            # if first iteration, query everything
            query_indices = np.arange(sd_features.shape[0])
        
        else:
            query_indices = query_generator.get_query_indices(
                indices=np.arange(samples.shape[0]), 
                query_type=cfg.query_conf.query_type,
                # n_queries=cfg.query_conf.n_feedbacks_per_query,
                **query_kwargs,
            )

        human_rewards = feedback_interface.query_batch(
            prompts=["amusement street"]*query_indices.shape[0],
            image_batch=samples[query_indices],
            query_indices=np.arange(query_indices.shape[0]),
        )
        human_rewards = torch.tensor(human_rewards)
        max_reward_in_batch, index = torch.max(human_rewards, 0)
        best_index = query_indices[index]
        if max_reward_in_batch > best_reward:
            best_image = samples[best_index].cpu().numpy().transpose(1, 2, 0) * 255
            best_image = PIL.Image.fromarray(best_image.astype('uint8'), mode='RGB')
            best_sample_latent = sd_features[best_index].unsqueeze(0)
            best_noise_latent = sd_noises[best_index].unsqueeze(0)
            best_reward = max_reward_in_batch
            best_image.save("best_image.png")
        # print("Got human rewards for query indices", query_indices)
        print("human rewards", human_rewards)
        
        # add to human dataset
        human_dataset.add_data(
            sd_features=sd_features[query_indices],
            human_rewards=human_rewards,
            ai_rewards=torch.zeros_like(human_rewards),  # A hack for unused AI rewards
        )
        print(f"Added {human_rewards.shape[0]} human data to human dataset. There are {human_dataset.n_data} data in human dataset")
    
        # update number of human feedback collected
        n_total_human_feedback += human_rewards.shape[0]

        ############################ If we have enough human data to train on ############################

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        # get human encodings for samples in this batch
        with torch.no_grad():
            scores = human_dataset.human_rewards
            positive_indices = torch.nonzero(scores > scores.mean()).squeeze()
            positive_features = human_dataset.sd_features[positive_indices]

        print("\nComputing final rewards")
        ### Similarity to the best sample ### (for similarity-based reward)
        if cfg.ddpo_conf.reward_mode == "similarity-to-best-sample":
            final_rewards = torch.nn.functional.cosine_similarity(sd_features, best_sample_latent.expand(sd_features.shape))
            final_rewards = (final_rewards+1)/2
        ### Similarity to all positive samples ###
        elif cfg.ddpo_conf.reward_mode == "similarity-to-all-positive":
            expand_dim = (positive_features.shape[0], sd_features.shape[0], sd_features.shape[1])
            positive_features = positive_features.unsqueeze(1).expand(expand_dim)
            sd_features = sd_features.unsqueeze(0).expand(expand_dim)
            final_rewards = torch.nn.functional.cosine_similarity(sd_features, positive_features, dim=2).mean(0)
        ### Ground truth Pickscore reward ###
        elif cfg.ddpo_conf.reward_mode == "pickscore":
            final_rewards = torch.zeros_like(human_rewards)
            for i, q_index in enumerate(query_indices):
                # final_rewards[q_index] = human_rewards[i]
                final_rewards[q_index] = human_rewards[i] - 19.2
        else:
            print("\n")

        human_dataset.clear_data()

        ai_indices = np.setdiff1d(np.arange(sd_features.shape[0]), np.array(query_indices)) # feature indices where AI reward is used

        # if using dummy human, get feedback for samples queried to the human for evaluation
        if (ai_indices.shape[0] > 0) and cfg.query_conf.feedback_agent == "ai":
            ############################################################
            # Get final rewards for DDPO training
            ############################################################
            # get ground truth human rewards
            gnd_truth_human_rewards = feedback_interface.query_batch(
                prompts=["amusement street"]*ai_indices.shape[0],
                image_batch=samples[ai_indices],
                query_indices=np.arange(ai_indices.shape[0]),
            )
            gnd_truth_human_rewards = torch.Tensor(gnd_truth_human_rewards)

        print("Got final rewards", final_rewards)
        
        ############################################################
        # Train SD via DDPO
        ############################################################
        print("Training DDPO...")
        ddpo_trainer.train_from_reward_labels(
            raw_rewards=final_rewards,
            logger=logger,
            epoch=loop,
        )
        n_ddpo_train_calls += 1

        # increment loop
        loop += 1

        # log
        if query_indices.shape[0] > 0:
            if ai_indices.shape[0] > 0:
                # both human and ai were queried
                mean_human_reward = torch.cat((human_rewards, gnd_truth_human_rewards)).mean()
            else:
                # only human was queried
                mean_human_reward = human_rewards.mean()
        else:
            # only ai was queried
            mean_human_reward = gnd_truth_human_rewards.mean()

        accelerator.log({
            "human_feedback_total" : n_total_human_feedback,
            "n_ddpo_training" : n_ddpo_train_calls,
            "mean_human_reward_this_batch" : mean_human_reward,
            "best_score_so_far": best_reward,
        })


if __name__ == "__main__":
    # app.run(main)
    main()