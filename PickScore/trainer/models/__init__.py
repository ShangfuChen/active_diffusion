from hydra.core.config_store import ConfigStore

from PickScore.trainer.models.clip_model import ClipModelConfig
from PickScore.trainer.models.pickscore_model import PickScoreModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="clip", node=ClipModelConfig)
cs.store(group="model", name="pickscore", node=PickScoreModelConfig)


