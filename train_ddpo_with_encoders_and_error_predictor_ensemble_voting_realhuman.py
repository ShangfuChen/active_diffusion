

import hydra

from accelerate import Accelerator
from accelerate.logging import get_logger
from PickScore.trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from accelerate.utils import set_seed, ProjectConfiguration


import os
import numpy as np
import torch
import random
import datetime

from ddpo_trainer import DDPOTrainer
from rl4dgm.user_feedback_interface.preference_functions import ColorPickOne, ColorScoreOne, PickScore
from rl4dgm.user_feedback_interface.user_feedback_interface import HumanFeedbackInterface, AIFeedbackInterface
from rl4dgm.utils.query_generator import EvaluationQueryGenerator
from rl4dgm.models.mydatasets import TripletDataset, DoubleTripletDataset, FeatureDoubleLabelDataset, HumanDataset
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer, DoubleTripletEncoderTrainer
from rl4dgm.reward_predictor_trainers.reward_predictor_trainers import ErrorPredictorEnsembleTrainerVoting

logger = get_logger(__name__)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# MODIFYING FOR HUMAN FEEDBACK #
###### Main training loop ######
@hydra.main(version_base=None, config_path="PickScore/trainer/conf", config_name="config")
def main(cfg: TrainerConfig) -> None:

    print("-"*50)
    print("Config", cfg)
    print("\n\n", cfg.dataset.dataset_name)
    print("-"*50)
    
    assert cfg.query_conf.feedback_agent == "human", "Make sure feedback agent is set to human"

    # create directories to save sampled images
    img_save_dir = os.path.join("/home/hayano/sampled_images", cfg.ddpo_conf.run_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=False)
    
    
    n_human_data_for_training = cfg.human_encoder_conf.n_data_needed_for_training # minimum number of human data to accumulate before training the human encoder
    is_enough_human_data = False

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
    print("Initializing error predictor trainer...")
    reward_predictor_trainer = ErrorPredictorEnsembleTrainerVoting(
        trainset=None,
        testset=None,
        config_dict=dict(cfg.error_predictor_ensemble_conf),
        seed=cfg.ddpo_conf.seed,
        accelerator=accelerator,
        save_dir=os.path.join(cfg.ddpo_conf.save_dir, cfg.ddpo_conf.run_name, "reward_predictor"),
    )
    print("...done\n")
    torch.manual_seed(cfg.ddpo_conf.seed) # DONT REMOVE THIS
    
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
        save_dir=os.path.join(cfg.ddpo_conf.save_dir, cfg.ddpo_conf.run_name, "ai_encoder"),
    )
    print("...done\n")

    print("Initializing human encoder trainer...")
    human_encoder_trainer = DoubleTripletEncoderTrainer(
        config_dict=dict(cfg.human_encoder_conf),
        seed=cfg.ddpo_conf.seed,
        trainset=None,
        testset=None,
        accelerator=accelerator,
        save_dir=os.path.join(cfg.ddpo_conf.save_dir, cfg.ddpo_conf.run_name, "human_encoder"),
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
    if cfg.query_conf.query_type == "random":
        query_kwargs = {"n_queries" : cfg.query_conf.n_feedback_per_query}
    elif cfg.query_conf.query_type == "ensemble_std":
        if cfg.query_conf.ensemble_thresh_is_hard:
            query_kwargs = {"thresh" : cfg.query_conf.ensemble_std_thresh}
        else:
            # if using dunamic thresholding, initialize at zero
            query_kwargs = {"thresh" : 0.0}
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

    n_ddpo_train_calls = 0 # number of times ddpo training loop was called
    loop = 0 # number of times the main training loop has run
    
    while n_ddpo_train_calls < cfg.ddpo_conf.n_outer_loops:
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
        print("ai rewards", ai_rewards)
        
        if ai_reward_max is None or ai_rewards.max() > ai_reward_max:
            ai_reward_max = ai_rewards.max()
        if ai_reward_min is None or ai_rewards.min() < ai_reward_min:
            ai_reward_min = ai_rewards.min()

        # update number of AI feedback collected
        n_total_ai_feedback += ai_rewards.shape[0]

        sd_features = torch.flatten(all_latents[:,-1,:,:,:], start_dim=1).float()
        print("Sampled from SD model and flattened featuers to ", sd_features.shape)

        ############################################################
        # Active query --> get human rewards
        ############################################################
        print("Begin active query")
        if loop == 0 and cfg.query_conf.query_everything_fisrt_iter:
            # if first iteration, query everything
            query_indices = np.arange(sd_features.shape[0])
            n_active_query_indices = query_indices.shape[0]
        
        else:
            # choose indices to query
            if cfg.query_conf.query_type == "ensemble_std":
                human_encodings = human_encoder_trainer.model(sd_features)
                ai_encodings = ai_encoder_trainer.model(sd_features)
                human_and_ai_encodings = torch.cat([human_encodings, ai_encodings], dim=1)
                ensemble_out, ensemble_mean, ensemble_std = reward_predictor_trainer.get_emsemble_out_mean_std(human_and_ai_encodings)
                query_kwargs["stds"] = ensemble_std.cpu()
                print("stds", ensemble_std)
                accelerator.log({
                    "ensemble_std" : ensemble_std.mean(),
                })
                if not cfg.query_conf.ensemble_thresh_is_hard:
                    std_thresh = cfg.query_conf.ensemble_dynamic_std_thresh * (human_reward_max - human_reward_min)
                    query_kwargs["thresh"] = std_thresh
                    print("std thresh", std_thresh)
                    accelerator.log({
                        "ensemble_std_thresh" : std_thresh,
                    })

            query_indices = query_generator.get_query_indices(
                indices=np.arange(samples.shape[0]), 
                query_type=cfg.query_conf.query_type,
                # n_queries=cfg.query_conf.n_feedbacks_per_query,
                **query_kwargs,
            )
            n_active_query_indices = query_indices.shape[0]
            
            # if still in warmup phase and the number of queries is too little, add random indices to query
            if query_indices.shape[0] < cfg.query_conf.min_n_queries:
                is_warmup = human_encoder_trainer.n_calls_to_train >= cfg.human_encoder_conf.n_warmup_epochs
                if is_warmup or not cfg.query_conf.only_enforce_min_queries_during_warmup:
                    n_random_queries = cfg.query_conf.min_n_queries - query_indices.shape[0]
                    indices = np.setdiff1d(np.arange(samples.shape[0]), query_indices)
                    additional_random_query_indices = query_generator.get_query_indices(
                        indices=indices,
                        query_type="random",
                        n_queries=n_random_queries,
                    )
                    query_indices = np.concatenate([query_indices, additional_random_query_indices])
                    print(f"not enough queries from active query. randomly sampled {n_random_queries}: {query_indices}")
            
        # get human rewards
        if query_indices.shape[0] > 0:
            human_rewards = feedback_interface.query_batch(
                prompts=["an aesthetic cat"]*query_indices.shape[0],
                image_batch=samples[query_indices],
                query_indices=np.arange(query_indices.shape[0]),
            )
            human_rewards = torch.tensor(human_rewards).float()

            print("Got human rewards for query indices", query_indices)
            print("human rewards", human_rewards)

            if human_reward_max is None or human_rewards.max() > human_reward_max:
                human_reward_max = human_rewards.max()
            if human_reward_min is None or human_rewards.min() < human_reward_min:
                human_reward_min = human_rewards.min()
            
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
        
        # decide if human encoder and reward predictor should be trained this loop
        is_enough_human_data = human_dataset.n_data >= n_human_data_for_training


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
        

        ############################################################
        # Train human encoder
        ############################################################
        if is_enough_human_data:
            # get up-to-date human and AI encodings for the features in the human dataset
            with torch.no_grad():
                re_encoded_human_features = human_encoder_trainer.model(human_dataset.sd_features)
                re_encoded_ai_features = ai_encoder_trainer.model(human_dataset.sd_features)
            
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
            
        # get encodings from human encoder
        with torch.no_grad():
            human_encodings = human_encoder_trainer.model(sd_features)
        
        # concatenate human and ai encodings for this batch
        human_and_ai_encodings = torch.cat([human_encodings, ai_encodings], dim=1)

        # if using real human feedback, make reward prediction before training predictor on the new data (for test accuracy)
        if cfg.query_conf.feedback_agent == "human":
            with torch.no_grad():
                real_human_prediction = reward_predictor_trainer.model(human_and_ai_encodings)

        ############################################################
        # Train error predictor
        ############################################################
        if is_enough_human_data:
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
            reward_predictor_trainer.initialize_dataloaders(trainset=error_predictor_trainset)
            reward_predictor_trainer.train_model()

            # clear the human_dataset
            human_dataset.clear_data()

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        print("\nComputing final rewards")
        final_rewards = torch.zeros(sd_features.shape[0])
        ai_indices = np.setdiff1d(np.arange(sd_features.shape[0]), np.array(query_indices)) # feature indices where AI reward is used

        # get error predictions
        with torch.no_grad():
            predictions = reward_predictor_trainer.model(human_and_ai_encodings)

        # correct the ai feedback
        corrected_ai_rewards = ai_rewards[ai_indices].cpu() + predictions[ai_indices].cpu()
        print("Original AI rewards", ai_rewards[ai_indices])
        print("Corrected AI rewards", corrected_ai_rewards)

        # compute final rewards
        final_rewards[ai_indices] = corrected_ai_rewards
        final_rewards[query_indices] = human_rewards
        print("Got final rewards", final_rewards)

        ############################################################
        # Train SD via DDPO
        ############################################################
        if human_encoder_trainer.n_calls_to_train > cfg.human_encoder_conf.n_warmup_epochs:            
            assert reward_predictor_trainer.n_calls_to_train == human_encoder_trainer.n_calls_to_train, \
                f"number of train calls to human encoder {human_encoder_trainer.n_calls_to_train} and error predictor {reward_predictor_trainer.n_calls_to_train} does not match!"
            print("Training DDPO...")
            ddpo_trainer.train_from_reward_labels(
                raw_rewards=final_rewards,
                logger=logger,
                epoch=loop,
            )
            n_ddpo_train_calls += 1
        else:
            print(f"Human encoder and error predictor warmup has not completed. Skipping DDPO training. Warmup step {human_encoder_trainer.n_calls_to_train}/{cfg.human_encoder_conf.n_warmup_epochs}")

        # increment loop
        loop += 1


        ############################################################
        # Evaluation and logging
        ############################################################
        # if using dummy human, get ground truth feedback for samples not queried to the human for evaluation
        if (ai_indices.shape[0] > 0) and cfg.query_conf.feedback_agent == "ai": # TODO - not tested
            # get ground truth human rewards
            gnd_truth_human_rewards = feedback_interface.query_batch(
                prompts=["an aesthetic cat"]*ai_indices.shape[0],
                image_batch=samples[ai_indices],
                query_indices=np.arange(ai_indices.shape[0]),
            )
            gnd_truth_human_rewards = torch.Tensor(gnd_truth_human_rewards)
            print("gnd truth human rewards", gnd_truth_human_rewards)

            human_reward_prediction_error = torch.abs(gnd_truth_human_rewards - corrected_ai_rewards) 
            human_reward_prediction_percent_error = human_reward_prediction_error / (human_reward_max - human_reward_min)

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
                "loop": loop,
                "human_reward_prediction_mean_error" : human_reward_prediction_error.mean(),
                "human_reward_prediction_mean_percent_error" : human_reward_prediction_percent_error.mean(),
                "under_10%_error_ratio" : (human_reward_prediction_percent_error < 0.1).sum() / ai_indices.shape[0],
                "under_20%_error_ratio" : (human_reward_prediction_percent_error < 0.2).sum() / ai_indices.shape[0],
                "under_30%_error_ratio" : (human_reward_prediction_percent_error < 0.3).sum() / ai_indices.shape[0],
                "human_feedback_total" : n_total_human_feedback,
                "ai_feedback_total" : n_total_ai_feedback,
                "n_active_queries" : n_active_query_indices,
                "n_ai_encoder_training" : ai_encoder_trainer.n_calls_to_train,
                "n_human_encoder_training" : human_encoder_trainer.n_calls_to_train,
                "n_reward_predictor_training" : reward_predictor_trainer.n_calls_to_train,
                "n_ddpo_training" : n_ddpo_train_calls,
                "mean_human_reward_this_batch" : mean_human_reward,
            }) 


        if cfg.query_conf.feedback_agent == "human": # TODO : need check for case where human was not queried
            # compute reward prediction accuracy
            real_human_reward_prediction_error = torch.abs(human_rewards - (ai_rewards[query_indices].cpu() + real_human_prediction[query_indices].cpu()))
            real_human_reward_prediction_percent_error = real_human_reward_prediction_error / 10 # TODO - change if needed

            accelerator.log({
                "loop": loop,
                "human_reward_prediction_mean_error" : real_human_reward_prediction_error.mean(),
                "human_reward_prediction_mean_percent_error" : real_human_reward_prediction_percent_error.mean(),
                "under_10%_error_ratio" : (real_human_reward_prediction_percent_error < 0.1).sum() / query_indices.shape[0],
                "under_20%_error_ratio" : (real_human_reward_prediction_percent_error < 0.2).sum() / query_indices.shape[0],
                "under_30%_error_ratio" : (real_human_reward_prediction_percent_error < 0.3).sum() / query_indices.shape[0],
                "human_feedback_total" : n_total_human_feedback,
                "ai_feedback_total" : n_total_ai_feedback,
                "n_active_queries" : n_active_query_indices,
                "n_ai_encoder_training" : ai_encoder_trainer.n_calls_to_train,
                "n_human_encoder_training" : human_encoder_trainer.n_calls_to_train,
                "n_reward_predictor_training" : reward_predictor_trainer.n_calls_to_train,
                "n_ddpo_training" : n_ddpo_train_calls,
                "mean_human_reward_this_batch" : human_rewards.mean(),
            }) 
        # log
        
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