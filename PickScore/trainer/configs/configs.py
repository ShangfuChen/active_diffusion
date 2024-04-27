from dataclasses import dataclass, field
from typing import List, Any, Dict

from omegaconf import DictConfig, MISSING

import PickScore.trainer.accelerators
import PickScore.trainer.tasks
import PickScore.trainer.models
import PickScore.trainer.criterions
import PickScore.trainer.datasetss
import PickScore.trainer.optimizers
import PickScore.trainer.lr_schedulers
import PickScore.trainer.ddpo_conf
import PickScore.trainer.query_conf
from PickScore.trainer.accelerators.base_accelerator import BaseAcceleratorConfig
from PickScore.trainer.models.base_model import BaseModelConfig
from PickScore.trainer.tasks.base_task import BaseTaskConfig


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def instantiate_with_cfg(cfg: DictConfig, **kwargs):
    target = _locate(cfg._target_)
    # print("target", target)s
    # breakpoint()
    return target(cfg, **kwargs)


defaults = [
    # {"accelerator": "deepspeed"},
    {"accelerator": "debug"},
    {"task": "clip"},
    # {"model": "clip"},
    {"model": "pickscore"},
    {"criterion": "clip"},
    {"dataset": "my_dataset"},
    # {"optimizer": "dummy"},
    {"optimizer": "adamw"},
    # {"lr_scheduler": "dummy"},
    {"lr_scheduler": "constant_with_warmup"},
    {"ddpo_conf": "ddpo_train"},
    {"query_conf" : "query_conf"},
]


@dataclass
class DebugConfig:
    activate: bool = False
    port: int = 5900


@dataclass
class TrainerConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    accelerator: BaseAcceleratorConfig = MISSING
    task: BaseTaskConfig = MISSING
    model: BaseModelConfig = MISSING
    criterion: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING
    ddpo_conf: Any = MISSING
    query_conf: Any = MISSING
    debug: DebugConfig = DebugConfig()
    output_dir: str = "outputs"
