
import random
import torch 
from torch.utils.data import Dataset
from torch.distributions import Normal

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
        self.features = features.to(device)
        self.scores = scores.to(device)
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
            
            positive_idx = random.choices(population=self.indices, weights=positive_weights, k=1)[0].item()
            negative_idx = random.choices(population=self.indices, weights=negative_weights, k=1)[0].item()

            positive_feature = self.features[positive_idx]
            negative_feature = self.features[negative_idx]

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
        sampling_method="default",
    ):
        self.features = features.to(device)
        self.encoded_features = encoded_features.to(device)
        self.scores_self = scores_self.to(device)        
        self.scores_other = scores_other.to(device)
        self.indices = torch.arange(scores_self.shape[0]).to(device)
        self.score_range_self = scores_self.max() - scores_self.min()
        self.score_range_other = scores_other.max() - scores_other.min()
        self.sampling_std_self = sampling_std_self
        self.sampling_std_other = sampling_std_other
        self.sampling_method = sampling_method

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

            # sample positive and negative samples using distance with other encoding
            score_diff = self.scores_other[self.indices] - anchor_score
            most_to_least_similar_indices = torch.argsort(torch.abs(score_diff))
            positive_index = random.choice(most_to_least_similar_indices[1:int(0.1*self.features.shape[0])])
            positive_feature_other = self.features[positive_index]

            negative_index = random.choice(most_to_least_similar_indices[int(0.9*self.features.shape[0]):])
            negative_feature_other = self.features[negative_index]

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
            
            positive_idx = random.choices(population=self.indices, weights=positive_weights, k=1)[0].item()
            negative_idx = random.choices(population=self.indices, weights=negative_weights, k=1)[0].item()

            positive_feature_self = self.features[positive_idx]
            negative_feature_self = self.features[negative_idx]

            # sample positive and negative samples using distance with other encoding
            prob_density = torch.exp(dist.log_prob(self.scores_other))
            max_prob = prob_density.max()
            positive_weights = prob_density.clone() # sampling weights for positive samples
            positive_weights[torch.argmax(prob_density).item()] = 0.0 # exclude anchor
            negative_weights = max_prob - prob_density # sampling weights for negative samples
            
            positive_idx = random.choices(population=self.indices, weights=positive_weights, k=1)[0].item()
            negative_idx = random.choices(population=self.indices, weights=negative_weights, k=1)[0].item()

            positive_feature_other = self.features[positive_idx]
            negative_feature_other = self.features[negative_idx]

        return anchor_feature, anchor_score, positive_feature_self, negative_feature_self, positive_feature_other, negative_feature_other


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
    

