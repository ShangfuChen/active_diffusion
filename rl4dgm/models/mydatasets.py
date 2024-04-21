
import random
import torch 
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(
        self,
        features,
        scores,
        device,
        is_train=False,
    ):
        self.features = features.to(device)
        self.scores = scores.to(device)
        self.is_train = is_train
        self.indices = torch.arange(scores.shape[0]).to(device)
        self.score_range = scores.max() - scores.min()

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        # print("getitem ", idx)
        # given index, return anchor, positive, and negative samples
        # get anchor sample
        anchor_feature = self.features[idx]
        anchor_score = self.scores[idx]

        # TODO - sample with dynamic probability depending on score similarity
        score_diff = self.scores[self.indices] - anchor_score
        most_to_least_similar_indices = torch.argsort(torch.abs(score_diff))
        positive_index = random.choice(most_to_least_similar_indices[1:int(0.1*self.features.shape[0])])
        positive_feature = self.features[positive_index]

        negative_index = random.choice(most_to_least_similar_indices[int(0.9*self.features.shape[0]):])
        negative_feature = self.features[negative_index]
        
        # # get positive sample
        # positive_indices = self.indices[(self.indices != idx) \
        #     & (((self.scores[self.indices] - anchor_score)) < 0.2*self.score_range)]
        # positive_index = random.choice(positive_indices)
        # positive_feature = self.features[positive_index]
        
        # # get negative sample
        # negative_indices = self.indices[(self.scores[self.indices] - anchor_score) > 0.5*self.score_range]
        # negative_index = random.choice(negative_indices)
        # negative_feature = self.features[negative_index]

        return anchor_feature, anchor_score, positive_feature, negative_feature, 

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
    

