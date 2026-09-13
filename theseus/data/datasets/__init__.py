from .dataset import (
    Dataset,
    DatasetComponent,
    DatasetConfig,
    ChatTemplate,
    ChatTurn,
    ChatTemplateDataset,
    StringDataset,
    StreamingDataset,
    StreamingStringDataset,
    StreamingChatTemplateDataset,
    PretrainingDataset,
    StreamingPretrainingDataset,
    ContrastiveChatTemplateDataset,
    ContrastiveStringDataset,
    ContrastiveDataset,
)

# Import all dataset modules to trigger @dataset decorator registration.
from .alpaca import Alpaca  # noqa: F401
from .addition import Addition  # noqa: F401
from .bbq import BBQ  # noqa: F401
from .ccaligned import CCAligned  # noqa: F401
from .cfq import CFQ  # noqa: F401
from .clutrr import CLUTRR  # noqa: F401

from .fever import FEVER  # noqa: F401
from .fineweb import FineWeb  # noqa: F401
from .fineweb_edu import FineWebEduDedup  # noqa: F401
from .flan import Flan  # noqa: F401
from .harmfulqa import HarmfulQA  # noqa: F401
from .longbench import LongBench  # noqa: F401
from .longhealth import LongHealth  # noqa: F401
from .mmlu import MMLU  # noqa: F401
from .mnli import MNLI  # noqa: F401
from .mtob import MTOB  # noqa: F401
from .openr1_math import OpenR1Math  # noqa: F401
from .pes2o import Pes2O  # noqa: F401
from .pg19 import PG19  # noqa: F401
from .pile import Pile  # noqa: F401
from .pile_detoxify import PileDetoxify  # noqa: F401
from .pile_injected import PileInjected  # noqa: F401
from .qqp import QQP  # noqa: F401

from .siqa import SIQA  # noqa: F401
from .squad import SQuAD  # noqa: F401
from .sst2 import SST2  # noqa: F401
from .winogrande import Winogrande  # noqa: F401

__all__ = [
    "Dataset",
    "DatasetComponent",
    "DatasetConfig",
    "ChatTemplate",
    "ChatTurn",
    "ChatTemplateDataset",
    "StringDataset",
    "StreamingDataset",
    "StreamingStringDataset",
    "StreamingChatTemplateDataset",
    "PretrainingDataset",
    "StreamingPretrainingDataset",
    "ContrastiveChatTemplateDataset",
    "ContrastiveDataset",
    "ContrastiveStringDataset",
    "Alpaca",
    "Addition",
    "BBQ",
    "CCAligned",
    "CFQ",
    "CLUTRR",
    "FEVER",
    "FineWeb",
    "FineWebEduDedup",
    "Flan",
    "HarmfulQA",
    "LongBench",
    "LongHealth",
    "MMLU",
    "MNLI",
    "MTOB",
    "OpenR1Math",
    "Pes2O",
    "PG19",
    "Pile",
    "PileDetoxify",
    "PileInjected",
    "QQP",
    "SIQA",
    "SQuAD",
    "SST2",
    "Winogrande",
]
