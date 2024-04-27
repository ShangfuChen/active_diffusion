

import hydra

from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoModel
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

import os
import numpy as np
import torch
import datetime

from ddpo_trainer import DDPOTrainer
from rl4dgm.user_feedback_interface.preference_functions import ColorPickOne, ColorScoreOne, PickScore
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import EvaluationQueryGenerator
from rl4dgm.models.mydatasets import TripletDataset, DoubleTripletDataset, FeatureDoubleLabelDataset
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer, DoubleTripletEncoderTrainer
from rl4dgm.reward_predictor_trainers.reward_predictor_trainers import ErrorPredictorTrainer

logger = get_logger(__name__)


###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)

    # create directories to save sampled images
    img_save_dir = os.path.join("/home/hayano/sampled_images", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    
    
    ############################################
    # Initialize trainers
    ############################################
    ddpo_trainer = DDPOTrainer(config=cfg.ddpo_conf, logger=logger, accelerator=None)
    ai_encoder_trainer = TripletEncoderTrainer(
        config_dict = { # TODO - put this in config
            "batch_size" : 16,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 2,
            "triplet_margin" : 1.0,
            "save_dir" : None,
            "save_every" : 50,
            "input_dim" : 32768,
            "n_hidden_layers" : 3,
            "hidden_dims" : [32768, 8192, 2048, 512],
            "output_dim" : 512,
        },
        device=ddpo_trainer.accelerator.device,
        trainset=None,
        testset=None,
    )
    human_encoder_trainer = DoubleTripletEncoderTrainer(
        config_dict={ # TODO put this in config
            "batch_size" : 16,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 2,
            "save_dir" : None,
            "save_every" : 50,
            "agent1_triplet_margin" : 1.0,
            "agent2_triplet_margin" : 1.0,
            "agent1_loss_weight" : 1.0,
            "agent2_loss_weight" : 0.25,
            "input_dim" : 32768,
            "n_hidden_layers" : 3,
            "hidden_dims" : [32768, 8192, 2048, 512],
            "output_dim" : 512,
        },
        trainset=None,
        testset=None,
    )
    error_predictor_trainer = ErrorPredictorTrainer(
        trainset=None,
        testset=None,
        config_dict={ # TODO put this in config
            "batch_size" : 16,
            "shuffle" : True,
            "lr" : 1e-6,
            "n_epochs" : 2,
            "save_dir" : None,
            "save_every" : 50,
            "n_hidden_layers" : 3,
            "hidden_dims" : [512]*4,
            "output_dim" : 1,
        },
    )

    ############################################
    # Initialize other components
    ############################################
    query_generator = EvaluationQueryGenerator()

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

    high_reward_latents = None
    
    
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    for loop in range(60): # TODO put this in config

        ############################################################
        # Sample from SD model
        ############################################################
        samples, all_latents, prompts, ai_rewards = ddpo_trainer.sample(
        # samples, all_latents, prompts, ai_rewards = rlcm_trainer.sample(
            logger=logger,
            epoch=loop,
            save_images=True,
            img_save_dir=img_save_dir,
            high_reward_latents=None) 
            # high_reward_latents=high_reward_latents) 

        ai_rewards = [elem for sublist in ai_rewards for elem in sublist] # flatten list 
        print("check shape of all_latents")
        breakpoint()

        ############################################################
        # Active query --> get human rewards
        ############################################################
        # choose indices to query
        query_indices = query_generator.get_query_indices(
            indices=np.arange(samples.shape[0]), 
            query_type=cfg.query_conf.query_type,
            n_queries=cfg.query_conf.n_feedbacks_per_query,
        )
        # get human rewards
        human_rewards = feedback_interface.query_batch(
            prompts=["an aesthetic cat"]*query_indices.shape[0],
            image_batch=samples[query_indices],
            query_indices=np.arange(query_indices.shape[0]),
        )
        
        ############################################################
        # Train AI encoder
        ############################################################
        # make a dataset
        ai_encoder_trainset = TripletDataset(
            features=all_latents,
            scores=ai_rewards,
            device=DDPOTrainer.accelerator.device,
            sampling_std=0.2, # TODO add to confg
            smapling_method="default", # TODO add to config
        )
        ai_encoder_trainer.initialize_dataloaders(trainset=ai_encoder_trainset)
        ai_encoder_trainer.train_model()
        
        # get encodings from newly trained ai encoder
        ai_encodings = ai_encoder_trainer.model(all_latents[query_indices])
        
        ############################################################
        # Train human encoder
        ############################################################
        # make a dataset
        human_encoder_trainset = DoubleTripletDataset(
            features=all_latents[query_indices],
            encoded_features=ai_encodings,
            scores_self=human_rewards,
            scores_other=ai_rewards[query_indices],
            device=ddpo_trainer.accelerator.device,
            sampling_std_self=0.2, # TODO put these in config
            sampling_std_other=0.2,
            score_percent_error_thresh=0.1,
            sampling_method="default",
        )
        human_encoder_trainer.initialize_dataloaders(trainset=human_encoder_trainset)
        human_encoder_trainer.train_model()

        # get encodings using newly trained human encoder
        human_encodings = human_encoder_trainer.model(all_latents[query_indices])

        ############################################################
        # Train error predictor
        ############################################################
        # make a dataset
        human_and_ai_encodings = torch.cat([human_encodings, ai_encodings], dim=1)
        error_predictor_trainset = FeatureDoubleLabelDataset(
            features=human_and_ai_encodings,
            agent1_labels=human_rewards,
            agent2_labels=ai_rewards[query_indices],
            device=ddpo_trainer.accelerator.device,
        )
        error_predictor_trainer.initialize_dataloaders(trainset=error_predictor_trainset)
        error_predictor_trainer.train_model()

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        final_rewards = torch.zeros(all_latents.shape[0])
        final_rewards[query_indices] = human_rewards 
        ai_indices = np.setdiff1d(np.array(query_indices.cpu()), np.arange(all_latents.shape[0])) # feature indices where AI reward is used
        predicted_error = error_predictor_trainer.model(all_latents[ai_indices]) 
        final_rewards[ai_indices] = ai_rewards[ai_indices] + predicted_error

        ############################################################
        # Train SD via DDPO
        ############################################################
        ddpo_trainer.train_from_reward_labels(
            raw_rewards=final_rewards,
            logger=logger,
            epoch=loop,
        )


        # # Batch normalization
        # if cfg.ddpo_conf.normalization:
        #     human_rewards = np.array(human_rewards)
        #     human_rewards = (human_rewards - human_rewards.min())/(human_rewards.max() - human_rewards.min()).tolist()
        #     ai_rewards = np.array(ai_rewards)
        #     ai_rewards = (ai_rewards - ai_rewards.min())/(ai_rewards.max() - ai_rewards.min()).tolist()
        
        # ddpo_trainer.accelerator.log(
        #     {"AI reward": np.asarray(ai_rewards).mean(),
        #      "Human reward": np.asarray(human_rewards).mean()}
        # )

        
        # ddpo_trainer.train_from_reward_labels(logger=logger, epoch=loop, raw_rewards=final_rewards)


if __name__ == "__main__":
    # app.run(main)
    main()