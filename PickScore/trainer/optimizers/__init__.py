from hydra.core.config_store import ConfigStore

from PickScore.trainer.optimizers.adamw import AdamWOptimizerConfig
from PickScore.trainer.optimizers.dummy_optimizer import DummyOptimizerConfig

cs = ConfigStore.instance()
cs.store(group="optimizer", name="dummy", node=DummyOptimizerConfig)
cs.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)
