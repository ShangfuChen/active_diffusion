from hydra.core.config_store import ConfigStore
from PickScore.trainer.human_encoder_conf.human_encoder_config import HumanEncoderConfig

cs = ConfigStore.instance()
cs.store(group="human_encoder_conf", name="human_encoder_conf", node=HumanEncoderConfig)
