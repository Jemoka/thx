# Evaluations are registered via @evaluation decorators in their definition modules.
# See theseus.registry for the authoritative EVALUATIONS dict.
#
# Import all evaluation modules to trigger decorator registration.
from .alpaca import AlpacaEval  # noqa: F401
from .addition import AdditionEval  # noqa: F401
from .arc_challenge import ARCChallengeEval  # noqa: F401
from .arithmetic import ArithmeticEval  # noqa: F401
from .bbh import BBHEval  # noqa: F401
from .bbq import BBQEval  # noqa: F401
from .blimp import Blimp  # noqa: F401
from .ccaligned import CCALIGNED_EVALS  # noqa: F401
from .cfq import CFQEval  # noqa: F401
from .clutrr import CLUTRREval  # noqa: F401

from .fever import FEVEREval  # noqa: F401
from .gsm8k import GSM8KEval  # noqa: F401
from .longbench import LongBench  # noqa: F401
from .longhealth import LongHealthEval  # noqa: F401
from .math import MathEval  # noqa: F401
from .mmlu import MMLUEval  # noqa: F401
from .hellaswag import HellaSwagEval  # noqa: F401
from .mnli import MNLIEval  # noqa: F401
from .mtob import MTOBEval  # noqa: F401
from .pes2o import Pes2OEval  # noqa: F401
from .pg19 import PG19Eval  # noqa: F401
import theseus.evaluation.datasets.pg19_lengthgen  # noqa: F401
from .pile import PileEval  # noqa: F401
from .pile_injected import PileInjectedEval  # noqa: F401
from .qqp import QQPEval  # noqa: F401

from .siqa import SIQAEval  # noqa: F401
from .squad import SQuADEval  # noqa: F401
from .sst2 import SST2Eval  # noqa: F401
from .tinystories import TinyStoriesEval  # noqa: F401
from .winogrande import WinograndeEval  # noqa: F401
from .perplexity_evals import (  # noqa: F401
    MNLIPerplexityEval,
    QQPPerplexityEval,
    SST2PerplexityEval,
    SIQAPerplexityEval,
    WinograndePerplexityEval,
    FineWebPerplexityEval,
)
