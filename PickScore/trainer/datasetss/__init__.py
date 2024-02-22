from hydra.core.config_store import ConfigStore

from trainer.datasetss.clip_hf_dataset import CLIPHFDatasetConfig, PickapicOnlySomeDatasetConfig
from trainer.datasetss.my_hf_dataset import MyHFDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="clip", node=CLIPHFDatasetConfig)
cs.store(group="dataset", name="pickapic_only_some", node=PickapicOnlySomeDatasetConfig)
cs.store(group="dataset", name="my_dataset", node=MyHFDatasetConfig)
