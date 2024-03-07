from hydra.core.config_store import ConfigStore

from trainer.models.clip_model import ClipModelConfig
from trainer.models.pickscore_model import PickScoreModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="clip", node=ClipModelConfig)
cs.store(group="model", name="pickscore", node=PickScoreModelConfig)


