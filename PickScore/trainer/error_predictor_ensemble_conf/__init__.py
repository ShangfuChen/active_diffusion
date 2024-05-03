from hydra.core.config_store import ConfigStore
from PickScore.trainer.error_predictor_ensemble_conf.error_predictor_ensemble_config import ErrorPredictorEnsembleConfig

cs = ConfigStore.instance()
cs.store(group="error_predictor_ensemble_conf", name="error_predictor_ensemble_conf", node=ErrorPredictorEnsembleConfig)
