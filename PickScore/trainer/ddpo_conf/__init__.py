from hydra.core.config_store import ConfigStore
from PickScore.trainer.ddpo_conf.ddpo_train import DDPOTrainConfig

cs = ConfigStore.instance()
cs.store(group="ddpo_conf", name="ddpo_train", node=DDPOTrainConfig)
