

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
from rl4dgm.reward_predictor_trainers.encoder_trainers import TripletEncoderTrainer, RepresentationTrainer
from transformers import AutoProcessor, AutoModel

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
    img_save_dir = os.path.join("/home/samchen/sampled_images", cfg.ddpo_conf.run_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
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
    # query_generator = EvaluationQueryGenerator(seed=cfg.ddpo_conf.seed)

    ############################################
    # Initialize feedback interfaces
    ############################################
    # latent_save_dir = os.path.join(cfg.ddpo_conf.logdir, cfg.ddpo_conf.run_name, "latents")
    # os.makedirs(latent_save_dir)

    # if cfg.query_conf.feedback_agent == "ai":
    #     feedback_interface = AIFeedbackInterface(
    #         feedback_type=cfg.query_conf.feedback_type,
    #         preference_function=PickScore,
    #     )
    # else:
    #     raise Exception("Only support AI feedback.")

    # setup query config
    # if cfg.query_conf.query_type == "random":
    #     query_kwargs = {"n_queries" : cfg.query_conf.n_feedback_per_query}
    # elif cfg.query_conf.query_type == "all":
    #     query_kwargs = {}
    
    #########################################################
    # MAIN TRAINING LOOP
    #########################################################
    n_total_human_feedback = 0

    n_ddpo_train_calls = 0 # number of times ddpo training loop was called
    loop = 0 # number of times the main training loop has run
    
    while n_ddpo_train_calls < cfg.ddpo_conf.n_outer_loops:
        ############################################################
        # Sample from SD model
        ############################################################
        print("sampling from random latent")
        samples, prompts = ddpo_trainer.sample(
            logger=logger,
            epoch=loop,
            save_images=True,
            img_save_dir=img_save_dir,
            high_reward_latents=None,
        )

        ############################################################
        # Active query --> get PickScore rewards
        ############################################################
        # print("Begin active query")
        # if loop == 0 and cfg.query_conf.query_everything_fisrt_iter:
        #     # if first iteration, query everything
        #     query_indices = np.arange(sd_features.shape[0])
        
        # else:
        #     query_indices = query_generator.get_query_indices(
        #         indices=np.arange(samples.shape[0]), 
        #         query_type=cfg.query_conf.query_type,
        #         # n_queries=cfg.query_conf.n_feedbacks_per_query,
        #         **query_kwargs,
        #     )
            
        #     # if still in warmup phase and the number of queries is too little, add random indices to query
        #     if query_indices.shape[0] < cfg.query_conf.min_n_queries:
        #         is_warmup = human_encoder_trainer.n_calls_to_train >= cfg.human_encoder_conf.n_warmup_epochs
        #         if is_warmup or not cfg.query_conf.only_enforce_min_queries_during_warmup:
        #             n_random_queries = cfg.query_conf.min_n_queries - query_indices.shape[0]
        #             indices = np.setdiff1d(np.arange(samples.shape[0]), query_indices)
        #             additional_random_query_indices = query_generator.get_query_indices(
        #                 indices=indices,
        #                 query_type="random",
        #                 n_queries=n_random_queries,
        #             )
        #             query_indices = np.concatenate([query_indices, additional_random_query_indices])
        #             print(f"not enough queries from active query. randomly sampled {n_random_queries}: {query_indices}")

        # feedback = feedback_interface.query_batch(
        #     prompts=[""]*query_indices.shape[0],
        #     image_batch=samples[query_indices],
        #     query_indices=np.arange(query_indices.shape[0]),
        # )
        # import ipdb
        # ipdb.set_trace()
        # if new_best_sample_index is not None:
        #     is_best_image_updated = True
        #     positive_indices = np.setdiff1d(positive_indices, new_best_sample_index)
        #     # if best sample was updated, update best_sample_latent
        #     best_sample_latent_prev = best_sample_latent
        #     best_sample_latent = sd_features[query_indices[new_best_sample_index]]
        #     best_noise_latent = sd_noises[query_indices[new_best_sample_index]].unsqueeze(0)
        # else:
        #     is_best_image_updated = False

        # positive_noise_latents = torch.cat([
        #     sd_noises[query_indices[positive_indices]],
        #     best_noise_latent,
        # ], dim=0)
    
        # # update number of human feedback collected
        # n_total_human_feedback += query_indices.shape[0]    

        # ############################################################
        # # Get final rewards for DDPO training
        # ############################################################
        # # get human encodings for samples in this batch
        # with torch.no_grad():
        #     human_encodings = human_encoder_trainer.model(sd_features)
        #     best_sample_encoding_from_prev_encoder = best_sample_encoding
        #     best_sample_encoding = human_encoder_trainer.model(best_sample_latent)
        #     if best_sample_encoding_from_prev_encoder is not None and not is_best_image_updated:
        #         best_to_best_cossim_cur_and_prev_encoder = torch.nn.functional.cosine_similarity(best_sample_encoding, best_sample_encoding_from_prev_encoder, dim=0).item()
        #         best_to_best_cossim_cur_and_prev_encoder = (best_to_best_cossim_cur_and_prev_encoder + 1) / 2
        #     if best_sample_latent_prev is not None and is_best_image_updated:
        #         best_sample_encoding_prev = human_encoder_trainer.model(best_sample_latent_prev)
        #         best_to_prev_best_cossim = torch.nn.functional.cosine_similarity(best_sample_encoding, best_sample_encoding_prev, dim=0).item()
        #         best_to_prev_best_cossim = (best_to_prev_best_cossim + 1) / 2
                
        # print("\nComputing final rewards")

        # if cfg.ddpo_conf.reward_mode == "similarity-to-best-sample":
        #     ### Similarity with the best sample ### (for similarity-based reward)
        #     final_rewards = torch.nn.functional.cosine_similarity(human_encodings, best_sample_encoding.expand(human_encodings.shape))
        #     final_rewards = (final_rewards+1)/2
        #     # final_rewards = torch.softmax(final_rewards, dim=0)
        # elif cfg.ddpo_conf.reward_mode == "similarity-to-all-positive":
        #     ### Similarity to all positive samples ###
        #     positive_sample_encodings = human_encodings[query_indices[positive_indices]]
        #     expand_dim = (positive_sample_encodings.shape[0], human_encodings.shape[0], human_encodings.shape[1])
        #     positive_sample_encodings = positive_sample_encodings.unsqueeze(1).expand(expand_dim)
        #     human_encodings = human_encodings.unsqueeze(0).expand(expand_dim)
        #     final_rewards = torch.nn.functional.cosine_similarity(human_encodings, positive_sample_encodings, dim=2).mean(0)
        #     final_rewards = (final_rewards+1)/2

        # ai_indices = np.setdiff1d(np.arange(sd_features.shape[0]), np.array(query_indices)) # feature indices where AI reward is used

        # # if using dummy human, get feedback for samples queried to the human for evaluation
        # if (ai_indices.shape[0] > 0) and cfg.query_conf.feedback_agent == "ai":
        #     ############################################################
        #     # Get final rewards for DDPO training
        #     ############################################################
        #     # get ground truth human rewards
        #     gnd_truth_human_rewards = feedback_interface.query_batch(
        #         prompts=["amusement street"]*ai_indices.shape[0],
        #         image_batch=samples[ai_indices],
        #         query_indices=np.arange(ai_indices.shape[0]),
        #     )
        #     gnd_truth_human_rewards = torch.Tensor(gnd_truth_human_rewards)

        # # print("Got final rewards", final_rewards)
        
        ############################################################
        # Train SD via DDPO
        ############################################################
        print("Training DDPO...")
        ddpo_trainer.train(
            logger=logger,
            epoch=loop,
            reward_model = AutoModel.from_pretrained(pretrained_model_name_or_path="yuvalkirstain/PickScore_v1").to("cuda").eval(),
            processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        )
        n_ddpo_train_calls += 1
        # increment loop
        loop += 1

if __name__ == "__main__":
    # app.run(main)
    main()