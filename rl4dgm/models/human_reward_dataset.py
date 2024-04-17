

from torch.utils.data import Dataset

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