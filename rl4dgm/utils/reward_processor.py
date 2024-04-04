
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
        
        DISTANCE_FUNCTIONS = {
            "l2" : self._compute_l2_distance,
        }
        SIMILARITY_FUNCTIONS = {
            "l2" : self._get_most_similar_l2,
        }

        assert distance_type in SIMILARITY_FUNCTIONS.keys(), f"distance_type must be one of {SIMILARITY_MEASURES.keys()}. Got {distance_type}."        
        self.distance_fn = DISTANCE_FUNCTIONS[distance_type]
        self.similarity_fn = SIMILARITY_FUNCTIONS[distance_type]
        # thresholds
        self.distance_thresh = distance_thresh
        self.reward_error_thresh = reward_error_thresh

        self.human_dataset = {
            "features" : [], # image features
            "noise_latents" : [], # image features
            "human_rewards" : [], # rewards from humans
            "ai_rewards" : [], # rewards from AI
            "reward_diff" : [], # Error of AI reward wrt human reward (R_human - R_AI)
        }

        self.total_n_human_feedback = 0
        self.total_n_ai_feedback = 0
        self.n_trusted_ai_feedback = 0
        self.n_corrected_ai_feedback = 0


    def compute_consensus_rewards(self, images, prompts, ai_rewards, feedback_interface, all_latents=None):        
        """
        Args:
            images (Tensor) : images to compute consensus rewards on. First dimension should be batch size
            features (Tensor) : image features to use for similarity computation. If not provided, images will be used instead
            prompts (list(str)) : list of prompts corresponding to each of the input images
            ai_rewards (list(float or int)) : AI rewards for the images in question
            feedback_interface (rl4dgm.user_feedback_interface.FeedbackInterface) : user feedback interface to query human
        """
        features = all_latents[:, -1, :, :, :]       
        noise_latents = all_latents[:, 0, :, :, :]
        batch_size = all_latents.shape[0]

        if features is None:
            features = images
        if isinstance(ai_rewards, list):
            ai_rewards = np.array(ai_rewards)

        print("AI rewards", ai_rewards)

        # get indices to query human and ai, along with indices of most representative samples for samples not queried to human
        human_query_indices, ai_query_indices, representative_sample_indices = self._active_query_by_similarity(features=features)

        self.total_n_human_feedback += human_query_indices.shape[0]
        self.total_n_ai_feedback += ai_query_indices.shape[0]

        print("\nhuman indices", human_query_indices)
        print("ai indices", ai_query_indices)
        print("representative indices", representative_sample_indices)

        ##### Aggregate human dataset #####
        human_rewards = feedback_interface.query_batch(prompts=prompts, image_batch=images, query_indices=human_query_indices)
        self.add_to_human_dataset(
            features=features[human_query_indices],
            noise_latents=noise_latents[human_query_indices],
            human_rewards=human_rewards,
            ai_rewards=ai_rewards[human_query_indices],
        )

        ##### Compute final rewards #####
        final_rewards = np.zeros(images.shape[0])
        # use human rewards where we have them
        final_rewards[human_query_indices] = np.array(human_rewards)

        if ai_query_indices.shape[0] > 0:
            # get error values from existing data
            errors = np.array(self.human_dataset["reward_diff"])[representative_sample_indices]
            # get input image indices where AI feedback is trustable. Use AI feedback directly for these samples
            trust_ai_indices = np.where(np.abs(errors) < self.reward_error_thresh)[0]
            trust_ai_indices = ai_query_indices[trust_ai_indices]
            final_rewards[trust_ai_indices] = ai_rewards[trust_ai_indices]

            # get input image indices where AI feedback is not trustable
            postprocess_ai_indices = np.where(np.abs(errors) >= self.reward_error_thresh)[0]
            errors_for_postprocessing = errors[postprocess_ai_indices]
            postprocess_ai_indices = ai_query_indices[postprocess_ai_indices]
            postprocessed_ai_rewards = ai_rewards[postprocess_ai_indices] + errors_for_postprocessing
            final_rewards[postprocess_ai_indices] = postprocessed_ai_rewards
            self.n_trusted_ai_feedback += trust_ai_indices.shape[0]
            self.n_corrected_ai_feedback += postprocess_ai_indices.shape[0]

            print("Trusted AI for indices", trust_ai_indices)
            print("Corrected AI for indices", postprocess_ai_indices)
            print("AI rewards before correction", ai_rewards[postprocess_ai_indices])
            print("Corrections", errors_for_postprocessing)
            print("Final rewards", final_rewards)

        print("Total human feedback:", self.total_n_human_feedback)
        print("Total AI feedback:", self.total_n_ai_feedback)
        print("Trusted AI feedback:", self.n_trusted_ai_feedback)
        print("Corrected AI feedback:", self.n_corrected_ai_feedback)

        # get idx with the top-{batch_size} human rewards
        high_reward_idx = np.argpartition(self.human_dataset["human_rewards"], -batch_size)[-batch_size:]
        high_reward_latents = torch.stack(
            [self.human_dataset["noise_latents"][i] for i in high_reward_idx],
            dim = 0)
        return final_rewards, high_reward_latents


    def add_to_human_dataset(self, features, noise_latents, human_rewards, ai_rewards):
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
        self.human_dataset["noise_latents"] += [latent for latent in noise_latents]
        self.human_dataset["human_rewards"] += human_rewards
        self.human_dataset["ai_rewards"] += ai_rewards
        error = np.array(human_rewards) - np.array(ai_rewards)
        self.human_dataset["reward_diff"] += error.tolist()



    ##################################################
    # DISTANCE METRICS #
    ##################################################

    def _get_most_similar_l2(self, features1, features2, nth_smallest=1):
        """
        Computes L2 distance between each feature in features1 to all features in features2
        Args:
            features1 (Tensor) : features or images
            features2 (Tensor) : features or images to compare features in feature1 to
            nth_smallest (int) : nth_smallest = 1 returns the most similar, n_thsmallest = 2 returns the second most similar, etc.
        Returns:
            min_distances (list(float)) : minimum distance between each input (in features1) to the most similar image in features2
            most_similar_data_indices (list(int)) : indices of most similar images [idx of feature in features2 most similart to features1[0], ....]
        """

        distances = self._compute_l2_distance(features1, features2)
        most_similar = distances.min(axis=1)
        min_distances = most_similar.values
        most_similar_data_indices = most_similar.indices

        return np.array(min_distances), np.array(most_similar_data_indices)

        # nth_smallest_distances = []
        # nth_most_similar_data_indices = []
        # # human_dataset_features = torch.stack([f for f in self.human_dataset["features"]])

        # for feature in features1:
        #     # compute distance to all images in the human dataset
        #     # dists = torch.cdist(feature, features, p=2.0).mean(dim=tuple(np.arange(1, features.dim()))) # something is wrong with this
        #     # TODO - hardcoded scaling factor 100
        #     dists = (features2 - feature).pow(2).mean(dim=tuple(np.arange(1, features1.dim())))
        #     # save the smallest distance
        #     nth_similar_data_idx = torch.argsort(dists)[nth_smallest - 1]
        #     nth_smallest_dist = dists[nth_similar_data_idx]
        #     nth_smallest_distances.append(nth_smallest_dist.item())
        #     nth_most_similar_data_indices.append(nth_similar_data_idx.item())

        #     # smallest_dist = dists.min(dim=0)
        #     # min_distances.append(smallest_dist.values.item())
        #     # most_similar_data_indices.append(smallest_dist.indices.item())
            
        # return np.array(nth_smallest_distances), np.array(nth_most_similar_data_indices)

    def _compute_l2_distance(self, features1, features2):
        """
        Computes L2 distance between each feature in features1 to all features in features2
        Args:
            features1 (Tensor) : features or images
            features2 (Tensor) : features or images to compare features in feature1 to
            nth_smallest (int) : nth_smallest = 1 returns the most similar, n_thsmallest = 2 returns the second most similar, etc.
        Returns:
            distances (tensor) : each row is a distance from sample i to all other samples including itself
        """

        distances = []
        for f in features1:
            dists = (features2 - f).pow(2).mean(dim=tuple(np.arange(1, features1.dim())))
            distances.append(dists.tolist())

        return torch.tensor(distances)

    
    ##################################################
    # ACTIVE QUERY ALGORITHMS #
    ##################################################

    def _active_query_by_similarity(self, features):
        """
        Args:
            features (Tensor) : features of sampled batch to use for similarity computation
        
        Returns:
            human_query_indices (np.array) : indices (of input features) to query human 
            ai_query_indices (np.array) : indices (of input features) to query AI
            representative_sample_indices (np.array) : indices (of sample in self.human_dataset) 
                that are most similar to each of the samples to be queried to AI
        """

        # When human data contains no data
        if len(self.human_dataset["human_rewards"]) == 0:
            human_query_indices = np.arange(features.shape[0])
            ai_query_indices = np.array([])
            representative_sample_indices = np.array([])
            return human_query_indices, ai_query_indices, representative_sample_indices

        # Compute similarity to samples in human dataset
        human_dataset_features = torch.stack([f for f in self.human_dataset["features"]])
        distances, most_similar_data_indices = self.similarity_fn(features, human_dataset_features)
        print("\nminimum distances computed", distances)

        # apply threshold and get indices of candidate samples to query
        # candidate_query_indices = np.array([0, 1, 2, 3, 4, 7])
        candidate_query_indices = np.where(distances > self.distance_thresh)[0] # indices for samples where similar sample exist in human dataset
        ai_query_indices = np.setdiff1d(np.arange(distances.shape[0]), candidate_query_indices)
        print("\ncandidate query indices", candidate_query_indices)
        # filter out redundant samples
        candidate_distances = np.array(self.distance_fn(features[candidate_query_indices], features[candidate_query_indices]))
        print("\ndistances among sample batch\n", candidate_distances)
        to_query = []
        not_to_query = []
        representative_sample_indices = []

        if candidate_query_indices.shape[0] > 1:
            is_similar = candidate_distances < self.distance_thresh
            priority_indices = np.argsort(is_similar.sum(axis=1))[::-1] # candidate indices ordered from most to least similar samples

            for idx in priority_indices:
                # get indices of similar samples
                similar_samples = np.where(is_similar[idx])[0]
                if idx not in to_query and idx not in not_to_query:
                    to_query.append(idx)
                    skip_samples = [s for s in similar_samples if s not in not_to_query and not s == idx]
                    not_to_query += skip_samples
                    representative_sample_indices += [candidate_query_indices[idx]] * len(skip_samples)
                
                # print("to_query", to_query)
                # print("not to query", not_to_query)
                # breakpoint()

            human_query_indices = candidate_query_indices[to_query]
            representative_sample_indices = np.concatenate([
                np.array(most_similar_data_indices, dtype=int)[ai_query_indices],
                np.array(representative_sample_indices, dtype=int) + len(self.human_dataset["human_rewards"])
            ])
            ai_query_indices = np.concatenate([ai_query_indices, candidate_query_indices[not_to_query]])

        else:
            human_query_indices = candidate_query_indices
            representative_sample_indices = np.array(most_similar_data_indices, dtype=int)[ai_query_indices]

        return human_query_indices, ai_query_indices, representative_sample_indices

"""
human_dataset_features = torch.stack([f for f in self.human_dataset["features"]])

# should be 0s
(self.distance_fn(features[human_query_indices], human_dataset_features) < self.distance_thresh).sum(axis=1)

# ai_indices[indices where below == 0] are the redundant samples
(self.distance_fn(features[ai_query_indices], human_dataset_features) < self.distance_thresh).sum(axis=1)

# most_similar_indices should match representative_sample_indices
self.similarity_fn(features[ai_query_indices], human_dataset_features)

"""
