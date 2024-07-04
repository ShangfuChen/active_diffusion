from hydra.core.config_store import ConfigStore
from PickScore.trainer.ddpo_conf_inference.ddpo_inference import DDPOInferenceConfig

cs = ConfigStore.instance()
cs.store(group="ddpo_conf_inference", name="ddpo_inference", node=DDPOInferenceConfig)

