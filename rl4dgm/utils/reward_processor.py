
import numpy as np
import torch

from transformers import AutoProcessor, AutoModel

class RewardProcessor:
    """
    Stores data labeled by human, compares it to new queries, and compute consensus reward labels using AI and human labeled data
    """
    def __init__(
        self,
        distance_thresh=10.0,
        distance_type="l2",
        reward_error_thresh=2.0,
    ):
        """
        distance_thresh (float) : samples with similarity value above this is considered similar
        distance_type (str) : type of distance metric to use
        reward_error_thresh (float) : reward error threshold to use when determining whether AI feedback is trustable for a given sample
            if R_H - R_AI > reward_error_thresh, AI feedback for this sample will be postprocessed
        """
        
        SIMILARITY_MEASURES = {
            "l2" : self._compute_l2_distance,
        }

        assert distance_type in SIMILARITY_MEASURES.keys(), f"distance_type must be one of {SIMILARITY_MEASURES.keys()}. Got {distance_type}."        
        self.distance_fn = SIMILARITY_MEASURES[distance_type]
        
        # thresholds
        self.distance_thresh = distance_thresh
        self.reward_error_thresh = reward_error_thresh

        self.human_dataset = {
            "features" : [], # image features
            "human_rewards" : [], # rewards from humans
            "ai_rewards" : [], # rewards from AI
            "reward_diff" : [], # Error of AI reward wrt human reward (R_human - R_AI)
        }


    def compute_consensus_rewards(self, images, prompts, ai_rewards, feedback_interface, features=None):        
        """
        Args:
            images (Tensor) : images to compute consensus rewards on. First dimension should be batch size
            features (Tensor) : image features to use for similarity computation. If not provided, images will be used instead
            prompts (list(str)) : list of prompts corresponding to each of the input images
            ai_rewards (list(float or int)) : AI rewards for the images in question
            feedback_interface (rl4dgm.user_feedback_interface.FeedbackInterface) : user feedback interface to query human
        """
        if features is None:
            features = images
        if isinstance(ai_rewards, list):
            ai_rewards = np.array(ai_rewards)

        ##### Compute similarity and determine samples to query human for #####
        if len(self.human_dataset["human_rewards"]) > 0:
            distances, most_similar_data_indices = self.distance_fn(features)
            # apply threshold and get indices of samples to query
            query_indices = np.where(distances > self.distance_thresh)[0]
            no_query_indices = np.where(distances <= self.distance_thresh)[0]
            print("query indices", query_indices)
            print("no query indices", no_query_indices)

        else:
            print("Human dataset is empty. All samples will be queried to human evaluator.")
            query_indices = np.arange(images.shape[0])
            no_query_indices = None
        
        ##### Aggregate human dataset #####
        human_rewards = feedback_interface.query_batch(prompts=prompts, image_batch=images, query_indices=query_indices)
        self.add_to_human_dataset(
            features=features[query_indices],
            human_rewards=human_rewards,
            ai_rewards=ai_rewards[query_indices],
        )

        ##### Compute final rewards #####
        final_rewards = np.zeros(images.shape[0])
        # use human rewards where we have them
        final_rewards[query_indices] = np.array(human_rewards)

        if no_query_indices is not None:
            # get error values from existing data
            errors = np.array(self.human_dataset["reward_diff"])[most_similar_data_indices]
            # get input image indices where AI feedback is trustable. Use AI feedback directly for these samples
            trust_ai_indices = np.where(np.abs(errors) < self.reward_error_thresh)[0]
            trust_ai_indices = np.setdiff1d(trust_ai_indices, query_indices) # remove indices queried by humans
            final_rewards[trust_ai_indices] = ai_rewards[trust_ai_indices]

            # get input image indices where AI feedback is not trustable
            postprocess_ai_indices = np.where(np.abs(errors) >= self.reward_error_thresh)[0]
            postprocess_ai_indices = np.setdiff1d(postprocess_ai_indices, query_indices) # remove indices queried by humans
            postprocessed_ai_rewards = ai_rewards[postprocess_ai_indices] + errors[postprocess_ai_indices]
            final_rewards[postprocess_ai_indices] = postprocessed_ai_rewards

            print("Trusted AI for indices", trust_ai_indices)
            print("Corrected AI for indices", postprocess_ai_indices)
            print("AI rewards before correction", ai_rewards[postprocess_ai_indices])
            print("Corrections", errors[postprocess_ai_indices])
            print("most similar images", most_similar_data_indices)

        return final_rewards

    def add_to_human_dataset(self, features, human_rewards, ai_rewards):
        """
        Add new datapoints to human dataset. 
        
        Args:
            features (Tensor) : image features to add to the dataset. First dimension should be batch size
            human_rewards (list(float or int) or array(float or int)) : human rewards for the input images
            ai_rewards (list(float or int) or array(float or int)) : AI rewards for the input images
        """

        # Cast inputs to list
        if type(human_rewards) == np.ndarray:
            human_rewards = human_rewards.tolist()
        if type(ai_rewards) == np.ndarray:
            ai_rewards = ai_rewards.tolist()

        self.human_dataset["features"] += [feature for feature in features]
        self.human_dataset["human_rewards"] += human_rewards
        self.human_dataset["ai_rewards"] += ai_rewards
        error = np.array(human_rewards) - np.array(ai_rewards)
        self.human_dataset["reward_diff"] += error.tolist()



    ##################################################
    # DISTANCE METRICS #
    ##################################################

    def _compute_l2_distance(self, features):
        """
        Computes L2 distance between input images and each of the images in the human dataset to find the most similar sample
        Args:
            images (Tensor) : features or images to add to the dataset. First dimension should be batch size
        Returns:
            min_distances (list(float)) : minimum distance between each input to the most similar image in the human data sample
            most_similar_data_indices (list(int)) : 
        """

        min_distances = []
        most_similar_data_indices = []

        for feature in features:
            # compute distance to all images in the human dataset
            dists = torch.cdist(feature, features, p=2.0).mean(dim=tuple(np.arange(1, features.dim())))
            # save the smallest distance
            smallest_dist = dists.min(dim=0)
            min_distances.append(smallest_dist.values.item())
            most_similar_data_indices.append(smallest_dist.indices.item())

        return np.array(min_distances), np.array(most_similar_data_indices)

        


