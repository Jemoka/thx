from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling
from theseus.training.schedules import WSD
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.model.models import Thoughtbubbles

from theseus.registry import job


@job("thoughtbubbles/train/pretrain")
class PretrainThoughtbubbles(BaseTrainer[BaseTrainerConfig, Thoughtbubbles]):
    DATASET = Sampling(FineWeb, 1.0, "pmd")
    MODEL = Thoughtbubbles
    CONFIG = BaseTrainerConfig

    SCHEDULE = WSD
