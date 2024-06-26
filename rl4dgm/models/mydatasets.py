
import random
import torch 
import numpy as np
from torch.utils.data import Dataset
from torch.distributions import Normal

def get_weighted_sample(weights, k):
    """
    Given weights, sample an index with probability proportional to the weights
    Args:
        weights
        k (int) : number of samples to draw
    """
    weights_cumsum = torch.cumsum(weights, dim=0)
    m = weights_cumsum[-1]
    indices = []
    i = 0
    while i < k:
        rand = m * torch.rand(1).item()
        idx = torch.searchsorted(sorted_sequence=weights_cumsum, input=rand, side='right')
        if idx in indices:
            continue
        indices.append(idx)
        i +=1
    return torch.tensor(indices)


class TripletDatasetWithBestSample(Dataset):
    def __init__(
        self,
        features,
        best_sample_feature,
        scores,
        best_sample_score,
        device,
        is_train=False,
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        self.best_sample_feature = best_sample_feature.float().to(device)
        self.scores = scores.float().to(device)
        self.best_sample_score = best_sample_score.float().to(device)
        self.is_train = is_train
        self.indices = torch.arange(scores.shape[0]).to(device)
        self.score_range = scores.max() - scores.min()
        self.sampling_method = sampling_method
        
        sampling_methods = [
            "default",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):

        # print("getitem ", idx)
        # given index, return anchor, positive, and negative samples
        
        # get anchor sample
        anchor_feature = self.best_sample_feature.squeeze()
        anchor_score = self.best_sample_score

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            positive_indices = torch.nonzero(self.scores > self.scores.mean()).squeeze()
            negative_indices = torch.nonzero(self.scores < self.scores.mean()).squeeze()
            positive_index = random.choice(positive_indices)
            negative_index = random.choice(negative_indices)
            positive_feature = self.features[positive_index]
            negative_feature = self.features[negative_index]
        return anchor_feature, anchor_score, positive_feature, negative_feature

class TripletDatasetWithPositiveNegativeBest(Dataset):
    def __init__(
        self,
        features,
        positive_indices,
        negative_indices,
        best_sample_feature,
        device,
        is_train=False,
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        self.best_sample_feature = best_sample_feature.float().to(device)
        self.is_train = is_train
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices

        self.sampling_method = sampling_method
        sampling_methods = [
            "default",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):

        # print("getitem ", idx)
        # given index, return anchor, positive, and negative samples
        
        # get anchor sample
        anchor_feature = self.best_sample_feature.squeeze()

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            positive_index = random.choice(self.positive_indices)
            negative_index = random.choice(self.negative_indices)
            positive_feature = self.features[positive_index]
            negative_feature = self.features[negative_index]
        return anchor_feature, 0, positive_feature, negative_feature # 0 is dummy and not used

class TripletDatasetWithPositiveNegative(Dataset):
    def __init__(
        self,
        features,
        positive_indices,
        negative_indices,
        # best_sample_feature,
        device,
        is_train=False,
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        # self.best_sample_feature = best_sample_feature.float().to(device)
        self.is_train = is_train
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices

        self.sampling_method = sampling_method
        sampling_methods = [
            "default",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

    def __len__(self):
        # return self.features.shape[0]
        return len(self.positive_indices)

    def __getitem__(self, idx):

        # print("getitem ", idx)
        # given index, return anchor, positive, and negative samples
        
        # get anchor sample (anchor is always positive sample)
        anchor_index = self.positive_indices[idx]
        anchor_feature = self.features[anchor_index]
        # anchor_index = random.choice(self.positive_indices)
        # anchor_feature = self.

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            # choose positive feature from positive samples (excluding the anchor)
            positive_index = random.choice(
                np.setdiff1d(np.array(self.positive_indices), np.array([anchor_index]))
            )
            positive_feature = self.features[positive_index]

            # choose negative fature from negative samples
            negative_index = random.choice(self.negative_indices)
            negative_feature = self.features[negative_index]

        return anchor_feature, 0, positive_feature, negative_feature # 0 is dummy and not used

class TripletDataset(Dataset):
    def __init__(
        self,
        features,
        scores,
        device,
        sampling_std=0.2, # std of Gaussian when sampling positive and negative samples. ratio of range of scores
        is_train=False,
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        self.scores = scores.float().to(device)
        self.is_train = is_train
        self.indices = torch.arange(scores.shape[0]).to(device)
        self.score_range = scores.max() - scores.min()
        self.sampling_std = sampling_std
        self.sampling_method = sampling_method
        
        sampling_methods = [
            "default",
            "weighted_gaussian",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):

        # print("getitem ", idx)
        # given index, return anchor, positive, and negative samples
        
        # get anchor sample
        anchor_feature = self.features[idx]
        anchor_score = self.scores[idx]

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            score_diff = self.scores[self.indices] - anchor_score
            most_to_least_similar_indices = torch.argsort(torch.abs(score_diff))
            positive_index = random.choice(most_to_least_similar_indices[1:int(0.1*self.features.shape[0])])
            positive_feature = self.features[positive_index]
            negative_index = random.choice(most_to_least_similar_indices[int(0.9*self.features.shape[0]):])
            negative_feature = self.features[negative_index]

        ##############################################################################################
        # Sample from gaussian depending on similarity
        ##############################################################################################
        elif self.sampling_method == "weighted_gaussian":
            mean = anchor_score
            std = self.sampling_std * self.score_range
            dist = Normal(loc=mean, scale=std) # normal distribution centered around anchor
            prob_density = torch.exp(dist.log_prob(self.scores))
            max_prob = prob_density.max()
            
            positive_weights = prob_density.clone() # sampling weights for positive samples
            positive_weights[torch.argmax(prob_density).item()] = 0.0 # exclude anchor
            negative_weights = max_prob - prob_density # sampling weights for negative samples
            
            positive_idx = get_weighted_sample(weights=positive_weights, k=1)
            negative_idx = get_weighted_sample(weights=negative_weights, k=1)

            positive_feature = torch.flatten(self.features[positive_idx], start_dim=0)
            negative_feature = torch.flatten(self.features[negative_idx], start_dim=0)

        return anchor_feature, anchor_score, positive_feature, negative_feature

class DoubleTripletDataset(Dataset):
    def __init__(
        self,
        features,
        encoded_features, # features encoded by another pretrained encoder that is different from the one we are training now
        scores_self,
        scores_other,
        device,
        sampling_std_self=0.2, # std of Gaussian when sampling positive and negative samples. ratio of range of scores
        sampling_std_other=0.2, # for choosing positive and negative samples according to encoded features. ratio of range of distances
        score_percent_error_thresh=0.1, 
        sampling_method="default",
    ):
        self.features = features.float().to(device)
        self.encoded_features = encoded_features.float().to(device)
        self.scores_self = scores_self.float().to(device)        
        self.scores_other = scores_other.float().to(device)
        self.indices = torch.arange(scores_self.shape[0]).to(device)
        self.score_range_self = scores_self.max() - scores_self.min()
        self.score_range_other = scores_other.max() - scores_other.min()
        self.sampling_std_self = sampling_std_self
        self.sampling_std_other = sampling_std_other
        self.score_percent_error_thresh = score_percent_error_thresh
        self.sampling_method = sampling_method
        self.device = device

        sampling_methods = [
            "default",
            "weighted_gaussian",
        ]
        assert sampling_method in sampling_methods, f"Sampling method must be one of {sampling_methods}. Got {sampling_method}"

        # normalize the scores of the other agent
        scale = self.score_range_self / self.score_range_other
        self.scores_other = ((self.scores_other - self.scores_other.min()) * scale) + self.scores_self.min()


    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # print("getitem ", idx)
        # given index, return anchor, positive (self), negative (self), positive (other), negative (other)
        
        # get anchor sample
        anchor_feature = self.features[idx]
        anchor_score = self.scores_self[idx]

        ##############################################################################################
        # Sample positive (and negative) samples randomly from top 10% most (least) similar samples
        ##############################################################################################
        
        if self.sampling_method == "default":
            # sample positive and negative features using self score 
            score_diff = self.scores_self[self.indices] - anchor_score
            most_to_least_similar_indices = torch.argsort(torch.abs(score_diff))
            
            positive_index = random.choice(most_to_least_similar_indices[1:int(0.1*self.features.shape[0])])
            positive_feature_self = self.features[positive_index]

            negative_index = random.choice(most_to_least_similar_indices[int(0.9*self.features.shape[0]):])
            negative_feature_self = self.features[negative_index]

        ##############################################################################################
        # Sample from gaussian depending on similarity
        ##############################################################################################
        elif self.sampling_method == "weighted_gaussian":
            # sample positive and negative features using self score 
            mean = anchor_score
            std = self.sampling_std_self * self.score_range_self
            dist = Normal(loc=mean, scale=std) # normal distribution centered around anchor
            
            prob_density = torch.exp(dist.log_prob(self.scores_self))
            max_prob = prob_density.max()
            positive_weights = prob_density.clone() # sampling weights for positive samples
            positive_weights[torch.argmax(prob_density).item()] = 0.0 # exclude anchor
            negative_weights = max_prob - prob_density # sampling weights for negative samples
            
            positive_index = get_weighted_sample(weights=positive_weights, k=1)
            negative_index = get_weighted_sample(weights=negative_weights, k=1)

            # positive_idx = random.choices(population=self.indices, weights=positive_weights, k=1)[0].item()
            # negative_idx = random.choices(population=self.indices, weights=negative_weights, k=1)[0].item()

            positive_feature_self = self.features[positive_index]
            negative_feature_self = self.features[negative_index]

        # get sample from other encoder
        other_feature = self.encoded_features[idx]
        score_diff = anchor_score - self.scores_other[idx]
        is_positive = (torch.abs(score_diff) / self.score_range_self < self.score_percent_error_thresh).item()

        return anchor_feature, anchor_score, positive_feature_self, negative_feature_self, other_feature, is_positive#, self.scores_other[idx]

class FeatureLabelDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        device,
    ):
        super(FeatureLabelDataset, self).__init__()
        self.features = features.float().to(device)
        self.labels = labels.float().to(device)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FeatureDoubleLabelDataset(Dataset):
    def __init__(
        self,
        features,
        agent1_labels, # labels from agent 1
        agent2_labels, # labels from agent 2
        device,
    ):
        super(FeatureDoubleLabelDataset, self).__init__()
        self.features = features.float().to(device)
        self.agent1_labels = agent1_labels.float().to(device)
        self.agent2_labels = agent2_labels.float().to(device)

        self.agent1_label_max = self.agent1_labels.max()
        self.agent1_label_min = self.agent1_labels.min()
        self.agent1_label_max = self.agent2_labels.max()
        self.agent1_label_min = self.agent2_labels.min()

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.agent1_labels[idx], self.agent2_labels[idx]
    

class HumanRewardDataset(Dataset):
    def __init__(
        self,
        features,
        human_rewards,
        ai_rewards,
        device,
    ):
        super(HumanRewardDataset, self).__init__()
        self.features = features.float().to(device)
        self.human_rewards = human_rewards.float().to(device)
        self.ai_rewards = ai_rewards.float().to(device)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.human_rewards[idx], self.ai_rewards[idx]
    

class HumanRewardDatasetNoImage(Dataset):
    def __init__(
        self,
        human_rewards,
        ai_rewards,
        device,
    ):
        super(HumanRewardDatasetNoImage, self).__init__()
        self.human_rewards = human_rewards.float().to(device)
        self.ai_rewards = ai_rewards.float().to(device)

    def __len__(self):
        return self.human_rewards.shape[0]
    
    def __getitem__(self, idx):
        return self.human_rewards[idx], self.ai_rewards[idx]
    

class HumanDataset():

    """
    Note : This class is used to accumulate human data while training the encoder-predictor pipelins
        It is NOT a torch dataset. 
    """
    def __init__(
        self, 
        n_data_to_accumulate, 
        device
    ):
        self.clear_data()

        self.n_data_to_accumulate = n_data_to_accumulate
        self.device = device
        
    def clear_data(self,):
        print("clear data called")
        self._sd_features = None
        self._human_rewards = None
        self._ai_rewards = None
        self._n_data = 0

    def add_data(self, sd_features, human_rewards, ai_rewards):
        """
        Add sd_features, human rewards, and ai rewards to the human dataset
        """
        # cast inputs to tensors, correct dimension, and put on correct device
        assert isinstance(sd_features, torch.Tensor), f"sd_features must be a torch.Tensor"
        sd_features = sd_features.to(self.device)
        sd_features = self._make_2d(sd_features)

        if not isinstance(human_rewards, torch.Tensor):
            human_rewards = torch.Tensor(human_rewards)
        human_rewards = human_rewards.to(self.device)

        if not isinstance(ai_rewards, torch.Tensor):
            ai_rewards = torch.Tensor(ai_rewards)
        ai_rewards = ai_rewards.to(self.device)

        # if this is the first set of data added, initialize
        if self._sd_features is None:
            self._sd_features = sd_features
            self._human_rewards = human_rewards
            self._ai_rewards = ai_rewards
        
        # otherwise concatenate new data
        else:
            self._sd_features = torch.cat([self._sd_features, sd_features], dim=0)
            self._human_rewards = torch.cat([self._human_rewards, human_rewards])
            self._ai_rewards = torch.cat([self._ai_rewards, ai_rewards])

        # update number of data
        self._n_data += sd_features.shape[0]

    def _make_2d(self, input_tensor):
        """
        If the input is 1d, add a dimension to make it 2d
        """
        if input_tensor.dim() == 1:
            return input_tensor[None,:]
        elif input_tensor.dim() == 2:
            return input_tensor
        else:
            raise Exception("input tensor dimension is incorrect")

    @property
    def n_data(self,):
        return self._n_data

    @property
    def sd_features(self,):
        return self._sd_features
    
    @property
    def human_rewards(self,):
        return self._human_rewards
    
    @property
    def ai_rewards(self,):
        return self._ai_rewards
    
# class HumanDatasetSimilarity():

#     """
#     Note : This class is used to accumulate human data while training the similarity-based pipeline
#         It is NOT a torch dataset. 
#     """
#     def __init__(
#         self, 
#         device
#     ):
#         self.clear_data()
#         self.device = device
#         self._positive_samples = None
#         self._negative_samples = None
        
#     def clear_data(self,):
#         print("clear data called")
#         self._sd_features = None
#         self._n_data = 0
#         self._positive_samples = None
#         self._negative_samples = None

#     def add_data(self, sd_features, positive_samples, negative_samples):
#         """
#         Add sd_features, human rewards, and ai rewards to the human dataset
#         """
#         # cast inputs to tensors, correct dimension, and put on correct device
#         assert isinstance(sd_features, torch.Tensor), f"sd_features must be a torch.Tensor"
#         sd_features = sd_features.to(self.device)
#         sd_features = self._make_2d(sd_features)

#         if not isinstance(positive_samples, torch.Tensor):
#             positive_samples = torch.Tensor(positive_samples)

#         if not isinstance(negative_samples, torch.Tensor):
#             negative_samples = torch.Tensor(negative_samples)

#         # if this is the first set of data added, initialize
#         if self._sd_features is None:
#             self._sd_features = sd_features
#             self._positive_sampels = positive_samples
#             self._negative_samples = negative_samples
        
#         # otherwise concatenate new data
#         else:
#             self._sd_features = torch.cat([self._sd_features, sd_features], dim=0)
#             self._positive_sampels = torch.cat([self._positive_sampels, positive_samples])
#             self._negative_samples = torch.cat([self._negative_samples, negative_samples])

#         # update number of data
#         self._n_data += sd_features.shape[0]

#     def _make_2d(self, input_tensor):
#         """
#         If the input is 1d, add a dimension to make it 2d
#         """
#         if input_tensor.dim() == 1:
#             return input_tensor[None,:]
#         elif input_tensor.dim() == 2:
#             return input_tensor
#         else:
#             raise Exception("input tensor dimension is incorrect")
    
#     @property
#     def positive_features(self):
#         return self._positive_samples
    
#     @property
#     def negative_features(self):
#         return self._negative_samples

#     @property
#     def n_data(self,):
#         return self._n_data

#     @property
#     def sd_features(self,):
#         return self._sd_features
    
#     @property
#     def is_positive(self):
#         return self._is_positive

class HumanDatasetSimilarity():

    """
    Note : This class is used to accumulate human data while training the similarity-based pipeline
        It is NOT a torch dataset. 
    """
    def __init__(
        self, 
        device
    ):
        self.clear_data()
        self.device = device
        
    def clear_data(self,):
        print("clear data called")
        self._sd_features = None
        self._n_data = 0
        self._positive_indices = None
        self._negative_indices = None

    def add_data(self, sd_features, positive_indices, negative_indices):
        """
        Add sd_features, human rewards, and ai rewards to the human dataset
        """
        # cast inputs to tensors, correct dimension, and put on correct device
        assert isinstance(sd_features, torch.Tensor), f"sd_features must be a torch.Tensor"
        sd_features = sd_features.to(self.device)
        sd_features = self._make_2d(sd_features)

        if not isinstance(positive_indices, np.ndarray):
            positive_indices = np.array(positive_indices)

        # if this is the first set of data added, initialize
        if self._sd_features is None:
            self._sd_features = sd_features
            self._positive_indices = positive_indices
            self._negative_indices = negative_indices
        
        # otherwise concatenate new data
        else:
            self._positive_indices = np.concatenate([self._positive_indices, positive_indices + self._sd_features.shape[0]])
            self._negative_indices = np.concatenate([self._negative_indices, negative_indices + self._sd_features.shape[0]])
            self._sd_features = torch.cat([self._sd_features, sd_features], dim=0)

        # update number of data
        self._n_data += sd_features.shape[0]

    def _make_2d(self, input_tensor):
        """
        If the input is 1d, add a dimension to make it 2d
        """
        if input_tensor.dim() == 1:
            return input_tensor[None,:]
        elif input_tensor.dim() == 2:
            return input_tensor
        else:
            raise Exception("input tensor dimension is incorrect")
    
    @property
    def positive_features(self):
        return self._sd_features[self._positive_indices]
    
    @property
    def negative_features(self):
        return self._sd_features[self._negative_indices]

    @property
    def n_data(self,):
        return self._n_data

    @property
    def sd_features(self,):
        return self._sd_features
    
    @property
    def positive_indices(self):
        return self._positive_indices
    
    @property
    def negative_indices(self):
        return self._negative_indices


 
 