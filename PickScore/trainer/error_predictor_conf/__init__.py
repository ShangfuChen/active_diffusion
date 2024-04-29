from hydra.core.config_store import ConfigStore
from PickScore.trainer.error_predictor_conf.error_predictor_config import ErrorPredictorConfig

cs = ConfigStore.instance()
cs.store(group="error_predictor_conf", name="error_predictor_conf", node=ErrorPredictorConfig)
