from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD

from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import GPT
from theseus.registry import job


@job("gpt/train/pretrain")
class PretrainGPT(BaseTrainer[BaseTrainerConfig, GPT]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = GPT
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD
