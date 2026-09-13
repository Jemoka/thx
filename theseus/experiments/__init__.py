# Jobs are registered via @job decorators in their definition modules.
# See theseus.registry for the authoritative JOBS dict.
from . import data as data  # noqa: F401
from .models.gpt import PretrainGPT  # noqa: F401

from .forking import PretrainThoughtbubbles  # noqa: F401

from .models.qwen import PretrainQwen, FinetuneBackboneQwen  # noqa: F401
from .models.qwen_3_5 import (  # noqa: F401
    PretrainQwen3_5,
    FinetuneBackboneQwen3_5,
    PretrainQwen3_5MoE,
    FinetuneBackboneQwen3_5MoE,
)
from .models.llama import PretrainLlama, FinetuneBackboneLlama  # noqa: F401
from .models.gpt_neox import PretrainGPTNeoX, FinetuneBackboneGPTNeoX  # noqa: F401


from .benchmark import *  # noqa: F401, F403
