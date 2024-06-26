from hydra.core.config_store import ConfigStore

from PickScore.trainer.criterions.clip_criterion import CLIPCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="clip", node=CLIPCriterionConfig)
