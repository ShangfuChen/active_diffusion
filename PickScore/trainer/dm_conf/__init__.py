from hydra.core.config_store import ConfigStore
from trainer.dm_conf.ddpo import DDPOTrainConfig
from trainer.dm_conf.rlcm import RLCMTrainConfig

cs = ConfigStore.instance()
cs.store(group="dm_conf", name="ddpo", node=DDPOTrainConfig)
cs.store(group="dm_conf", name="rlcm", node=RLCMTrainConfig)
