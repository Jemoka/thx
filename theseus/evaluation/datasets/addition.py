from dataclasses import dataclass
from typing import Any, Tuple

from theseus.config import configure, field, configuration, current_config
from omegaconf import OmegaConf
from theseus.data.datasets.addition import Addition
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation


@dataclass
class AdditionEvalConfig:
    split: str = field("eval/addition/split", default="test")
    generation_tokens: int = field("eval/addition/generation_tokens", default=20)


@evaluation("addition")
class AdditionEval(RolloutEvaluation):
    """Exact-match rollout evaluation at the addition ``<mid>`` boundary.

    Given the dataset sequence::

        <bos> 2 3 7 + 6 8 2 = <mid> 0 9 1 9 <eos>

    ``get`` returns the numeric IDs through ``<mid>`` as the prompt and the
    four digit IDs as the label. ``RolloutEvaluation`` removes generated token
    0 (``<eos>``) before exact comparison, so prediction and label have the
    same space-separated integer representation.
    """

    CONFIG = AdditionEvalConfig

    @classmethod
    def config(cls) -> list[type[Any]]:
        return [*super().config(), *Addition.config()]

    def __init__(self) -> None:
        self.args = configure(self.CONFIG)
        config = current_config()
        assert config is not None
        with configuration(
            OmegaConf.merge(config, {"data": {"addition": {"split": self.args.split}}})
        ):
            self.dataset = Addition()

    @property
    def name(self) -> str:
        return "addition"

    def __len__(self) -> int:
        return len(self.dataset)

    def max_new_tokens(self, inference: Any) -> int:
        del inference
        return int(self.args.generation_tokens)

    def get(self, indx: int) -> Tuple[str, str]:
        sequence = self.dataset[indx]
        prompt, boundary, answer = sequence.partition(f" {self.dataset.middle_token} ")
        eos_suffix = f" {self.dataset.eos_token}"
        if not boundary or not answer.endswith(eos_suffix):
            raise ValueError(f"malformed addition sequence at index {indx}")
        return f"{prompt} {self.dataset.middle_token}", answer.removesuffix(eos_suffix)

    def clean(self, y_hat: str) -> str:
        return y_hat.strip()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip() == y_hat.strip()
