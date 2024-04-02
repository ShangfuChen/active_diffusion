
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
            "human_rewards" : [], # rewards from humans
            "ai_rewards" : [], # rewards from AI
            "reward_diff" : [], # Error of AI reward wrt human reward (R_human - R_AI)
        }

        self.total_n_human_feedback = 0
        self.total_n_ai_feedback = 0
        self.n_trusted_ai_feedback = 0
        self.n_corrected_ai_feedback = 0


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

        print("AI rewards", ai_rewards)

        # get indices to query human and ai, along with indices of most representative samples for samples not queried to human
        human_query_indices, ai_query_indices, representative_sample_indices = self._active_query_by_similarity(features=features)

        self.total_n_human_feedback += human_query_indices.shape[0]
        self.total_n_ai_feedback += ai_query_indices.shape[0]

        ##### Aggregate human dataset #####
        human_rewards = feedback_interface.query_batch(prompts=prompts, image_batch=images, query_indices=human_query_indices)
        self.add_to_human_dataset(
            features=features[human_query_indices],
            human_rewards=human_rewards,
            ai_rewards=ai_rewards[human_query_indices],
        )

        ##### Compute final rewards #####
        final_rewards = np.zeros(images.shape[0])
        # use human rewards where we have them
        final_rewards[human_query_indices] = np.array(human_rewards)
        print("human indices", human_query_indices)
        print("ai indices", ai_query_indices)
        print("representative indices", representative_sample_indices)
        if ai_query_indices.shape[0] > 0:
            breakpoint()
            # get error values from existing data
            errors = np.array(self.human_dataset["reward_diff"])[representative_sample_indices]
            # get input image indices where AI feedback is trustable. Use AI feedback directly for these samples
            trust_ai_indices = np.where(np.abs(errors) < self.reward_error_thresh)[0]
            trust_ai_indices = np.setdiff1d(trust_ai_indices, human_query_indices) # remove indices queried by humans
            final_rewards[trust_ai_indices] = ai_rewards[trust_ai_indices]

            # get input image indices where AI feedback is not trustable
            postprocess_ai_indices = np.where(np.abs(errors) >= self.reward_error_thresh)[0]
            postprocess_ai_indices = np.setdiff1d(postprocess_ai_indices, human_query_indices) # remove indices queried by humans
            postprocessed_ai_rewards = ai_rewards[postprocess_ai_indices] + errors[postprocess_ai_indices]
            final_rewards[postprocess_ai_indices] = postprocessed_ai_rewards

            self.n_trusted_ai_feedback += trust_ai_indices.shape[0]
            self.n_corrected_ai_feedback += postprocess_ai_indices.shape[0]

            print("Trusted AI for indices", trust_ai_indices)
            print("Corrected AI for indices", postprocess_ai_indices)
            print("AI rewards before correction", ai_rewards[postprocess_ai_indices])
            print("Corrections", errors[postprocess_ai_indices])
            print("most similar images", most_similar_data_indices)
        
        print("Total human feedback:", self.total_n_human_feedback)
        print("Total AI feedback:", self.total_n_ai_feedback)
        print("Trusted AI feedback:", self.n_trusted_ai_feedback)
        print("Corrected AI feedback:", self.n_corrected_ai_feedback)

        return final_rewards

    # def compute_consensus_rewards(self, images, prompts, ai_rewards, feedback_interface, features=None):        
    #     """
    #     Args:
    #         images (Tensor) : images to compute consensus rewards on. First dimension should be batch size
    #         features (Tensor) : image features to use for similarity computation. If not provided, images will be used instead
    #         prompts (list(str)) : list of prompts corresponding to each of the input images
    #         ai_rewards (list(float or int)) : AI rewards for the images in question
    #         feedback_interface (rl4dgm.user_feedback_interface.FeedbackInterface) : user feedback interface to query human
    #     """
    #     if features is None:
    #         features = images
    #     if isinstance(ai_rewards, list):
    #         ai_rewards = np.array(ai_rewards)

    #     print("AI rewards", ai_rewards)

    #     ##### Compute similarity and determine samples to query human for #####
    #     if len(self.human_dataset["human_rewards"]) > 0:

    #         # #######################################
    #         # Compute similarity to samples in human dataset
    #         human_dataset_features = torch.stack([f for f in self.human_dataset["features"]])
    #         distances, most_similar_data_indices = self.similarity_fn(features, human_dataset_features)
    #         print("minimum distances computed", distances)

    #         # apply threshold and get indices of candidate samples to query
    #         # candidate_query_indices = np.where(distances > self.distance_thresh)[0]
    #         candidate_query_indices = np.arange(distances.shape[0])
    #         no_query_indices = np.setdiff1d(np.arange(distances.shape[0]), candidate_query_indices) # indices for samples where similar sample exist in human dataset

    #         # are there similar samples within the input batch?
    #         distances_self = self.distance_fn(features[candidate_query_indices], features[candidate_query_indices])
    #         self_indices = np.arange(distances.shape[0])
    #         to_query = [] # ones to query among the candidates
    #         not_to_query = [] # redundant ones among the candidates
    #         not_to_query_similar_indices = [] # which one is queried as a representative of the not_to_query indices?
    #         for i, d in enumerate(distances_self):
    #             similar_indices = self_indices[np.where(d <= self.distance_thresh)[0]]
    #             similar_indices = np.setdiff1d(similar_indices, np.array([i]))
    #             if i not in to_query and not np.isin(similar_indices, to_query).any():
    #                 # if none of the samples similar to this is scheduled to be queried, add to query
    #                 to_query.append(i)
    #                 # all other similar samples should not be queried
    #                 skip_indices = [idx for idx in similar_indices if idx not in not_to_query]
    #                 not_to_query += skip_indices
    #                 # record most similar image idx (excluding itself)
    #                 not_to_query_similar_indices += [i] * len(skip_indices)
                    
    #                 # if len(skip_indices) > 0:
    #                 #     not_to_query_similar_indices += np.argsort(np.array(distances_self[skip_indices]))[:,1].tolist()
    #                 # not_to_query_similar_indices += [candidate_query_indices[i]] * len(skip_indices)

    #         query_indices = candidate_query_indices[to_query]
    #         no_query_indices = np.concatenate([no_query_indices, candidate_query_indices[not_to_query]])
    #         # not_to_query_similar_indices = candidate_query_indices[not_to_query_similar_indices]
    #         # update most_similar_data_indices
    #         breakpoint()
    #         most_similar_data_indices = np.concatenate([
    #             most_similar_data_indices, 
    #             np.array(not_to_query_similar_indices) + len(self.human_dataset["human_rewards"])
    #         ])


    #         print("query indices", query_indices)
    #         print("no query indices", no_query_indices)
    #         # #######################################

    #     else:
    #         print("Human dataset is empty. All samples will be queried to human evaluator.")
    #         query_indices = np.arange(images.shape[0])
    #         no_query_indices = None
        
    #     self.total_n_human_feedback += query_indices.shape[0]

    #     ##### Aggregate human dataset #####
    #     human_rewards = feedback_interface.query_batch(prompts=prompts, image_batch=images, query_indices=query_indices)
    #     self.add_to_human_dataset(
    #         features=features[query_indices],
    #         human_rewards=human_rewards,
    #         ai_rewards=ai_rewards[query_indices],
    #     )

    #     ##### Compute final rewards #####
    #     final_rewards = np.zeros(images.shape[0])
    #     # use human rewards where we have them
    #     final_rewards[query_indices] = np.array(human_rewards)

    #     if no_query_indices is not None:

    #         self.total_n_ai_feedback += no_query_indices.shape[0]
            
    #         # get error values from existing data
    #         errors = np.array(self.human_dataset["reward_diff"])[most_similar_data_indices]
    #         # get input image indices where AI feedback is trustable. Use AI feedback directly for these samples
    #         trust_ai_indices = np.where(np.abs(errors) < self.reward_error_thresh)[0]
    #         trust_ai_indices = np.setdiff1d(trust_ai_indices, query_indices) # remove indices queried by humans
    #         final_rewards[trust_ai_indices] = ai_rewards[trust_ai_indices]

    #         # get input image indices where AI feedback is not trustable
    #         postprocess_ai_indices = np.where(np.abs(errors) >= self.reward_error_thresh)[0]
    #         postprocess_ai_indices = np.setdiff1d(postprocess_ai_indices, query_indices) # remove indices queried by humans
    #         postprocessed_ai_rewards = ai_rewards[postprocess_ai_indices] + errors[postprocess_ai_indices]
    #         final_rewards[postprocess_ai_indices] = postprocessed_ai_rewards

    #         self.n_trusted_ai_feedback += trust_ai_indices.shape[0]
    #         self.n_corrected_ai_feedback += postprocess_ai_indices.shape[0]

    #         print("Trusted AI for indices", trust_ai_indices)
    #         print("Corrected AI for indices", postprocess_ai_indices)
    #         print("AI rewards before correction", ai_rewards[postprocess_ai_indices])
    #         print("Corrections", errors[postprocess_ai_indices])
    #         print("most similar images", most_similar_data_indices)
        
    #     print("Total human feedback:", self.total_n_human_feedback)
    #     print("Total AI feedback:", self.total_n_ai_feedback)
    #     print("Trusted AI feedback:", self.n_trusted_ai_feedback)
    #     print("Corrected AI feedback:", self.n_corrected_ai_feedback)

    #     return final_rewards


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
        print("minimum distances computed", distances)

        # apply threshold and get indices of candidate samples to query
        # candidate_query_indices = np.array([0, 1, 2, 3, 4, 7])
        candidate_query_indices = np.where(distances > self.distance_thresh)[0] # indices for samples where similar sample exist in human dataset
        ai_query_indices = np.setdiff1d(np.arange(distances.shape[0]), candidate_query_indices)

        # filter out redundant samples
        candidate_distances = np.array(self.distance_fn(features[candidate_query_indices], features[candidate_query_indices]))
        print("distances among sample batch", candidate_distances)
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
                    representative_sample_indices += [idx] * len(skip_samples)
                
                # print("to_query", to_query)
                # print("not to query", not_to_query)
                # breakpoint()

            human_query_indices = candidate_query_indices[to_query]
            ai_query_indices = np.concatenate([ai_query_indices, candidate_query_indices[not_to_query]])
            representative_sample_indices = np.concatenate([
                np.array(most_similar_data_indices),
                np.array(representative_sample_indices) + len(self.human_dataset["human_rewards"])
            ])

        else:
            human_query_indices = candidate_query_indices
            representative_sample_indices = np.array([])

        return human_query_indices, ai_query_indices, representative_sample_indices
