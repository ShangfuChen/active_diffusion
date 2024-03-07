from hydra.core.config_store import ConfigStore
from trainer.query_conf.query_config import QueryConfig

cs = ConfigStore.instance()
cs.store(group="query_conf", name="query_conf", node=QueryConfig)
