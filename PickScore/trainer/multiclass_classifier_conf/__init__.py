from hydra.core.config_store import ConfigStore
from PickScore.trainer.multiclass_classifier_conf.multiclass_classifier_config import MulticlassClassifierConfig

cs = ConfigStore.instance()
cs.store(group="multiclass_classifier_conf", name="multiclass_classifier_conf", node=MulticlassClassifierConfig)
