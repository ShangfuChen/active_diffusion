from hydra.core.config_store import ConfigStore
from PickScore.trainer.encoder_conf.encoder_config import QueryConfig

cs = ConfigStore.instance()
cs.store(group="query_conf", name="query_conf", node=QueryConfig)
