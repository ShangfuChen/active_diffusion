

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
from rl4dgm.models.mydatasets import TripletDatasetWithPositiveNegativeBest, HumanDatasetSimilarity
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer

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
    img_save_dir = os.path.join("/home/hayano/sampled_images", cfg.ddpo_conf.run_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
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

    print("Initializing human encoder trainer...")
    human_encoder_trainer = TripletEncoderTrainer(
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
        feedback_interface = HumanFeedbackInterface(feedback_type="positive-indices")
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
    human_dataset = HumanDatasetSimilarity(
        device=accelerator.device,
    )
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    n_total_human_feedback = 0
    best_sample_latent = None
    best_reward = 0

    n_ddpo_train_calls = 0 # number of times ddpo training loop was called
    loop = 0 # number of times the main training loop has run
    
    while n_ddpo_train_calls < cfg.ddpo_conf.n_outer_loops:
        ############################################################
        # Sample from SD model
        ############################################################
        samples, all_latents, _, _ = ddpo_trainer.sample(
            logger=logger,
            epoch=loop,
            save_images=True,
            img_save_dir=img_save_dir,
            high_reward_latents=None)

        sd_features = torch.flatten(all_latents[:,-1,:,:,:], start_dim=1).float()
        print("Sampled from SD model and flattened featuers to ", sd_features.shape)

        # ############################################################
        # # Active query --> get human rewards
        # ############################################################
        print("Begin active query")
        if loop == 0 and cfg.query_conf.query_everything_fisrt_iter:
            # if first iteration, query everything
            query_indices = np.arange(sd_features.shape[0])
        
        else:
            query_indices = query_generator.get_query_indices(
                indices=np.arange(samples.shape[0]), 
                query_type=cfg.query_conf.query_type,
                # n_queries=cfg.query_conf.n_feedbacks_per_query,
                **query_kwargs,
            )
            
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

        new_best_sample_index, positive_indices, negative_indices = feedback_interface.query_batch(
            prompts=["amusement street"]*query_indices.shape[0],
            image_batch=samples[query_indices],
            query_indices=np.arange(query_indices.shape[0]),
        )
        if new_best_sample_index is not None:
            # if best sample was updated, update best_sample_latent
            best_sample_latent = sd_features[new_best_sample_index]

        # add to human dataset
        human_dataset.add_data(
            sd_features=sd_features[query_indices],
            positive_indices=positive_indices,
            negative_indices=negative_indices,
        )
        print(f"Added to human dataset. There are {human_dataset.n_data} data in human dataset")
    
        # update number of human feedback collected
        n_total_human_feedback += query_indices.shape[0]    
        
        print("n data", human_dataset.n_data)
        print("positive indices", human_dataset.positive_indices)
        print("negative indices", human_dataset.negative_indices)

        ############################ If we have enough human data to train on ############################
        if human_dataset.n_data >= n_human_data_for_training:
            ############################################################
            # Train human encoder
            ############################################################
            # make a dataset
            print("\nPreparing dataset for human encoder training")
            human_encoder_trainset = TripletDatasetWithPositiveNegativeBest(
                features=human_dataset.sd_features,
                best_sample_feature=best_sample_latent,
                device=accelerator.device,
                positive_indices=human_dataset.positive_indices,
                negative_indices=human_dataset.negative_indices,
                sampling_method="default", # TODO add to config
            )
            human_encoder_trainer.initialize_dataloaders(trainset=human_encoder_trainset)
            human_encoder_trainer.train_model()

            # clear the human_dataset
            human_dataset.clear_data()

        ############################ If we have enough human data to train on ############################

        ############################################################
        # Get final rewards for DDPO training
        ############################################################
        # get human encodings for samples in this batch
        with torch.no_grad():
            human_encodings = human_encoder_trainer.model(sd_features)
            best_sample_encoding = human_encoder_trainer.model(best_sample_latent)
        print("\nComputing final rewards")
        ### Similarity with the best sample ### (for similarity-based reward)
        final_rewards = torch.nn.functional.cosine_similarity(human_encodings, best_sample_encoding.expand(human_encodings.shape))
        final_rewards = (final_rewards+1)/2
        # final_rewards = torch.softmax(final_rewards, dim=0)

        ### Ground truth human reward ### (for query everything)
        # final_rewards = human_rewards

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
        if human_encoder_trainer.n_calls_to_train >= cfg.human_encoder_conf.n_warmup_epochs:            
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

        # # log
        # if query_indices.shape[0] > 0:
        #     if ai_indices.shape[0] > 0:
        #         # both human and ai were queried
        #         mean_human_reward = torch.cat((human_rewards, gnd_truth_human_rewards)).mean()
        #     else:
        #         # only human was queried
        #         mean_human_reward = human_rewards.mean()
        # else:
        #     # only ai was queried
        #     mean_human_reward = gnd_truth_human_rewards.mean()

        accelerator.log({
            "human_feedback_total" : n_total_human_feedback,
            "n_human_encoder_training" : human_encoder_trainer.n_calls_to_train,
            "n_ddpo_training" : n_ddpo_train_calls,
            # "mean_human_reward_this_batch" : mean_human_reward,
            # "best_score_so_far": best_reward,
        })


if __name__ == "__main__":
    # app.run(main)
    main()