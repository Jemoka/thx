from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models.contrib.qwen_3_5 import Qwen3_5
from theseus.model.models.contrib.qwen_3_5_moe import Qwen3_5MoE
from theseus.registry import job


@job("qwen_3_5/train/pretrain")
class PretrainQwen3_5(BaseTrainer[BaseTrainerConfig, Qwen3_5]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = Qwen3_5
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD


@job("qwen_3_5/train/finetune")
class FinetuneBackboneQwen3_5(BackbonedTrainer):
    """Finetune from a pretrained Qwen 3.5 (text-only) backbone.

    Config keys:
        architecture/backbone/implementation: "qwen_3_5"
        architecture/backbone/weights: e.g. "Qwen/Qwen3.5-0.8B"
    """

    DATASET = Sampling(FineWeb, 1.0, "pmd")

    SCHEDULE = WSD


@job("qwen_3_5_moe/train/pretrain")
class PretrainQwen3_5MoE(BaseTrainer[BaseTrainerConfig, Qwen3_5MoE]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = Qwen3_5MoE
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD


@job("qwen_3_5_moe/train/finetune")
class FinetuneBackboneQwen3_5MoE(BackbonedTrainer):
    """Finetune from a pretrained Qwen 3.5 MoE backbone.

    Config keys:
        architecture/backbone/implementation: "qwen_3_5_moe"
        architecture/backbone/weights: e.g. "Qwen/Qwen3.5-35B-A3B"
    """

    DATASET = Sampling(FineWeb, 1.0, "pmd")

    SCHEDULE = WSD
