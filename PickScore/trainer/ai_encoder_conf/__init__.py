from hydra.core.config_store import ConfigStore
from PickScore.trainer.ai_encoder_conf.ai_encoder_config import AIEncoderConfig

cs = ConfigStore.instance()
cs.store(group="ai_encoder_conf", name="ai_encoder_conf", node=AIEncoderConfig)
