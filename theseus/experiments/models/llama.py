from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import Llama
from theseus.registry import job


@job("llama/train/pretrain")
class PretrainLlama(BaseTrainer[BaseTrainerConfig, Llama]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = Llama
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD


@job("llama/train/finetune")
class FinetuneBackboneLlama(BackbonedTrainer):
    """Finetune from a pretrained Llama backbone.

    Config keys:
        architecture/backbone/implementation: "llama"
        architecture/backbone/weights: e.g. "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    """

    DATASET = Sampling(FineWeb, 1.0, "pmd")

    SCHEDULE = WSD
