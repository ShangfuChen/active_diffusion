

import hydra

from accelerate import Accelerator
from accelerate.logging import get_logger
from PickScore.trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from accelerate.utils import set_seed, ProjectConfiguration

import random
import os
import numpy as np
import torch
import datetime

from ddpo_trainer import DDPOTrainer
from rl4dgm.user_feedback_interface.preference_functions import ColorPickOne, ColorScoreOne, PickScore
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import EvaluationQueryGenerator
from rl4dgm.models.mydatasets import TripletDataset, DoubleTripletDataset, FeatureDoubleLabelDataset, HumanDataset
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer, DoubleTripletEncoderTrainer
from rl4dgm.reward_predictor_trainers.reward_predictor_trainers import ErrorPredictorTrainer

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
    img_save_dir = os.path.join("/home/shangfu/sampled_images", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    
    
    n_human_data_for_training = cfg.human_encoder_conf.n_data_needed_for_training # minimum number of human data to accumulate before training the human encoder
    
    # set seed
    np.random.seed(cfg.ddpo_conf.seed)
    torch.manual_seed(cfg.ddpo_conf.seed)
    random.seed(cfg.ddpo_conf.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.ddpo_conf.seed)
    torch.use_deterministic_algorithms(True)

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
    print("\nInitializing DDPO trainer...")
    ddpo_trainer = DDPOTrainer(
        config=cfg.ddpo_conf, 
        logger=logger, 
        accelerator=accelerator
    )
    print("...done\n")

    print("Initializing AI encoder trainer...")
    ai_encoder_trainer = TripletEncoderTrainer(
        config_dict=dict(cfg.ai_encoder_conf),
        seed=cfg.ddpo_conf.seed,
        trainset=None,
        testset=None,
        accelerator=accelerator,
    )
    print("...done\n")

    print("Initializing human encoder trainer...")
    human_encoder_trainer = DoubleTripletEncoderTrainer(
        config_dict=dict(cfg.human_encoder_conf),
        seed=cfg.ddpo_conf.seed,
        trainset=None,
        testset=None,
        accelerator=accelerator,
    )
    print("...done\n")
    
    print("Initializing error predictor trainer...")
    error_predictor_trainer = ErrorPredictorTrainer(
        trainset=None,
        testset=None,
        config_dict=dict(cfg.error_predictor_conf),
        seed=cfg.ddpo_conf.seed,
        accelerator=accelerator,
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
    high_reward_latents = None
    
    
    ############################################################
    # Initialize human dataset to accumulate
    ############################################################
    human_dataset = HumanDataset(
        n_data_to_accumulate=n_human_data_for_training,
        device=accelerator.device,
    )
    # maximum and minimum human and AI reward values observed so far
    human_reward_max, human_reward_min = None, None
    ai_reward_max, ai_reward_min = None, None
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    n_total_ai_feedback = 0
    n_total_human_feedback = 0

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
        ai_rewards = torch.tensor(ai_rewards)
        
        if ai_reward_max is None or ai_rewards.max() > ai_reward_max:
            ai_reward_max = ai_rewards.max()
        if ai_reward_min is None or ai_rewards.min() < ai_reward_min:
            ai_reward_min = ai_rewards.min()

        sd_features = torch.flatten(all_latents[:,-1,:,:,:], start_dim=1).float()
        print("Sampled from SD model and flattened featuers to ", sd_features.shape)

        ############################################################
        # Active query --> get human rewards
        ############################################################
        print("Begin active query")
        # calculate std using current error prediction models
        with torch.no_grad():
            ai_encodings = ai_encoder_trainer.model(sd_features)
            human_encodings = human_encoder_trainer.model(sd_features)
        human_and_ai_encodings = torch.cat([human_encodings, ai_encodings], dim=1)
        # stds = error_predictor_trainer.model.calculate_std(human_and_ai_encodings)
        
        # choose indices to query
        # if loop < 100:
        query_indices = query_generator.get_query_indices(
        indices=np.arange(samples.shape[0]),
        query_type="random",
        n_queries=cfg.query_conf.n_feedbacks_per_query,
        )
        # else:
        #     query_indices = query_generator.get_query_indices(
        #         indices=np.arange(samples.shape[0]), 
        #         query_type=cfg.query_conf.query_type,
        #         stds=stds.squeeze().detach().cpu(),
        #         ratio=0.3,
        #     )
        accelerator.log({"n_human_query" : query_indices.shape[0]})

        # get human rewards
        if query_indices.shape[0] > 0:
            human_rewards = feedback_interface.query_batch(
                prompts=["an aesthetic cat"]*query_indices.shape[0],
                image_batch=samples[query_indices],
                query_indices=np.arange(query_indices.shape[0]),
            )
            human_rewards = torch.tensor(human_rewards)

            print("Got human rewards for query indices", query_indices)

            if human_reward_max is None or human_rewards.max() > human_reward_max:
                human_reward_max = human_rewards.max()
            if human_reward_min is None or human_rewards.min() < human_reward_min:
                human_reward_min = human_rewards.min()
    
            if cfg.error_predictor_conf.model_type == "multiclass_classifier":
                #### TESTING - normalize to 1-10 
                scale = (9 - 0) / (human_reward_max - human_reward_min)
                human_rewards = ((human_rewards - human_reward_min) * scale) 
                human_rewards = torch.round(human_rewards)
                human_rewards = torch.clamp(human_rewards, 0, 9)
                #### TESTING - normalize to 1-10 
            
            # add to human dataset
            human_dataset.add_data(
                sd_features=sd_features[query_indices],
                human_rewards=human_rewards,
                ai_rewards=ai_rewards[query_indices],
            )
            print(f"Added {human_rewards.shape[0]} human data to human dataset. There are {human_dataset.n_data} data in human dataset")
        
            # update number of human feedback collected
            n_total_human_feedback += human_rewards.shape[0]
        else:
            print("No samples to query human for this batch")        
        
        ############################################################
        # Train AI encoder
        ############################################################
        # make a dataset
        print("\nPreparing dataset for AI encoder training")
        ai_encoder_trainset = TripletDataset(
            features=sd_features,
            scores=ai_rewards,
            device=accelerator.device,
            sampling_std=0.2, # TODO add to confg
            sampling_method="default", # TODO add to config
        )
        ai_encoder_trainer.initialize_dataloaders(trainset=ai_encoder_trainset)
        ai_encoder_trainer.train_model()
        
        # get encodings from newly trained ai encoder
        with torch.no_grad():
            ai_encodings = ai_encoder_trainer.model(sd_features)
        
        ############################ If we have enough human data to train on ############################
        if human_dataset.n_data >= n_human_data_for_training:

            # get up-to-date human and AI encodings for the features in the human dataset
            with torch.no_grad():
                re_encoded_human_features = human_encoder_trainer.model(human_dataset.sd_features)
                re_encoded_ai_features = ai_encoder_trainer.model(human_dataset.sd_features)
            
            ############################################################
            # Train human encoder
            ############################################################
            # make a dataset
            print("\nPreparing dataset for human encoder training")
            human_encoder_trainset = DoubleTripletDataset(
                features=human_dataset.sd_features,
                encoded_features=re_encoded_ai_features,
                scores_self=human_dataset.human_rewards,
                scores_other=human_dataset.ai_rewards,
                device=accelerator.device,
                sampling_std_self=0.2, # TODO put these in config
                sampling_std_other=0.2,
                score_percent_error_thresh=0.1,
                sampling_method="default",
            )
            human_encoder_trainer.initialize_dataloaders(trainset=human_encoder_trainset)
            human_encoder_trainer.train_model()

            ############################################################
            # Train error predictor
            ############################################################
            # make a dataset
            print("\nPreparing dataset for error predictor training")
            # get ai encodings for the features in the human data
            human_and_re_encoded_ai_features = torch.cat([re_encoded_human_features, re_encoded_ai_features], dim=1)
            error_predictor_trainset = FeatureDoubleLabelDataset(
                features=human_and_re_encoded_ai_features,
                agent1_labels=human_dataset.human_rewards,
                agent2_labels=human_dataset.ai_rewards,
                device=accelerator.device,
            )
            error_predictor_trainer.initialize_dataloaders(trainset=error_predictor_trainset)
            error_predictor_trainer.train_model()
            # error_predictor_trainer.train_model_ce()

            # clear the human_dataset
            human_dataset.clear_data()
        ############################ If we have enough human data to train on ############################

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        # get human encodings for samples in this batch

        with torch.no_grad():
            human_encodings = human_encoder_trainer.model(sd_features)
        human_and_ai_encodings = torch.cat([human_encodings, ai_encodings], dim=1)

        print("\nComputing final rewards")
        final_rewards = torch.zeros(sd_features.shape[0])
        final_rewards[query_indices] = human_rewards 
        ai_indices = np.setdiff1d(np.arange(sd_features.shape[0]), np.array(query_indices)) # feature indices where AI reward is used

        with torch.no_grad():
            predictions = error_predictor_trainer.model(human_and_ai_encodings)

        if ai_indices.shape[0] > 0:

            ############################################################
            # Get final rewards for DDPO training
            ############################################################
            # get ground truth human rewards
            gnd_truth_human_rewards = feedback_interface.query_batch(
                prompts=["an aesthetic cat"]*ai_indices.shape[0],
                image_batch=samples[ai_indices],
                query_indices=np.arange(ai_indices.shape[0]),
            )
            gnd_truth_human_rewards = torch.Tensor(gnd_truth_human_rewards)

            # record average human reward this entire batch 
            mean_human_reward = torch.cat((human_rewards, gnd_truth_human_rewards)).mean()

            # compare prediction to ground truth
            if cfg.error_predictor_conf.model_type == "multiclass_classifier":
                #### TESTING - normalize to 1-10, round and clip
                scale = (9 - 0) / (human_reward_max - human_reward_min)
                gnd_truth_human_rewards = ((gnd_truth_human_rewards - human_reward_min) * scale) 
                gnd_truth_human_rewards = torch.round(gnd_truth_human_rewards)
                gnd_truth_human_rewards = torch.clamp(gnd_truth_human_rewards, 0, 9)
                #### TESTING - normalize to 1-10, round and clip
                
                reward_prediction = (torch.argmax(predictions, dim=1)).cpu()
                final_rewards[ai_indices] = reward_prediction[ai_indices].float()
                
                human_reward_prediction_error = torch.abs(reward_prediction[ai_indices] - gnd_truth_human_rewards)
                human_reward_predicton_percent_error = human_reward_prediction_error / (human_reward_max - human_reward_min)
                print("predicted human rewards", reward_prediction)

            elif cfg.error_predictor_conf.model_type == "error_predictor" \
                or cfg.error_predictor_conf.model_type == "uncertainty_error_predictor":
                predictions = torch.flatten(predictions)
                corrected_ai_rewards = ai_rewards[ai_indices].cpu() + predictions[ai_indices].cpu()
                final_rewards[ai_indices] = corrected_ai_rewards
                print("Original AI rewards", ai_rewards[ai_indices])
                print("Corrected AI rewards", corrected_ai_rewards)

                human_reward_prediction_error = torch.abs(gnd_truth_human_rewards - corrected_ai_rewards) 
                human_reward_predicton_percent_error = human_reward_prediction_error / (human_reward_max - human_reward_min)

            print("gnd truth human rewards", gnd_truth_human_rewards)
            print("human rewards", human_rewards)

            accelerator.log({
                "n_corrected_feedback" : ai_indices.shape[0],
                "human_reward_prediction_mean_error" : human_reward_prediction_error.mean(),
                "human_reward_prediction_mean_percent_error" : human_reward_predicton_percent_error.mean(),
                "under_10%_error_ratio" : (human_reward_predicton_percent_error < 0.1).sum() / ai_indices.shape[0],
                "under_20%_error_ratio" : (human_reward_predicton_percent_error < 0.2).sum() / ai_indices.shape[0],
                "under_30%_error_ratio" : (human_reward_predicton_percent_error < 0.3).sum() / ai_indices.shape[0],
                "mean_human_reward_this_batch" : mean_human_reward,
            })

        print("Got final rewards", final_rewards)


        # if loop >= 10:
        ############################################################
        # Train SD via DDPO
        ############################################################
        print("Training DDPO...")
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
        
        # accelerator.log(
        #     {"AI reward": np.asarray(ai_rewards).mean(),
        #      "Human reward": np.asarray(human_rewards).mean()}
        # )

        
        # ddpo_trainer.train_from_reward_labels(logger=logger, epoch=loop, raw_rewards=final_rewards)


if __name__ == "__main__":
    # app.run(main)
    main()