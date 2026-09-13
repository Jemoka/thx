from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.training.backbone import BackbonedTrainer
from theseus.model.models import GPTNeoX
from theseus.registry import job


@job("gpt_neox/train/pretrain")
class PretrainGPTNeoX(BaseTrainer[BaseTrainerConfig, GPTNeoX]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = GPTNeoX
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD


@job("gpt_neox/train/finetune")
class FinetuneBackboneGPTNeoX(BackbonedTrainer):
    """Finetune from a pretrained GPT-NeoX/Pythia backbone.

    Config keys:
        architecture/backbone/implementation: "gpt_neox"
        architecture/backbone/weights: e.g. "EleutherAI/pythia-70m-deduped"
    """

    DATASET = Sampling(FineWeb, 1.0, "pmd")

    SCHEDULE = WSD
