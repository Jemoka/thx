from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import Qwen
from theseus.registry import job


@job("qwen/train/pretrain")
class PretrainQwen(BaseTrainer[BaseTrainerConfig, Qwen]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = Qwen
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD


@job("qwen/train/finetune")
class FinetuneBackboneQwen(BackbonedTrainer):
    """Finetune from a pretrained Qwen backbone.

    Config keys:
        architecture/backbone/implementation: "qwen"
        architecture/backbone/weights: e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    """

    DATASET = Sampling(FineWeb, 1.0, "pmd")

    SCHEDULE = WSD
