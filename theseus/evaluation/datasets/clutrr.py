from dataclasses import dataclass
from typing import Any, Tuple

from theseus.config import configure, field, configuration, current_config
from omegaconf import OmegaConf
from theseus.data.datasets import ChatTemplate
from theseus.data.datasets.clutrr import CLUTRR
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation


@dataclass
class CLUTRREvalConfig:
    split: str = field("eval/clutrr/split", default="test")
    system_prompt: str = field("eval/clutrr/system_prompt", default="")
    generation_tokens: int = field("eval/clutrr/generation_tokens", default=20)


@evaluation("clutrr")
class CLUTRREval(RolloutEvaluation):
    """CLUTRR relational reasoning evaluation using test split."""

    CONFIG = CLUTRREvalConfig

    @classmethod
    def config(cls) -> list[type[Any]]:
        return [*super().config(), *CLUTRR.config()]

    def __init__(self) -> None:
        self.args = configure(self.CONFIG)
        config = current_config()
        assert config is not None
        with configuration(
            OmegaConf.merge(config, {"data": {"clutrr": {"split": self.args.split}}})
        ):
            self.dataset = CLUTRR()
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "clutrr"

    def max_new_tokens(self, inference: Any) -> int:
        del inference
        return int(self.args.generation_tokens)

    def get(self, indx: int) -> Tuple[str, str]:
        turns = self.dataset[indx]
        prompt = encode_chat_template(
            turns[:-1],
            self.encoder,
            self.args.system_prompt,
            prompt=True,
            tokenize=False,
        )
        return prompt, turns[-1].message

    def __len__(self) -> int:
        return len(self.dataset)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        assistant_msgs = []
        for i in chats:
            if i.role == "assistant":
                assistant_msgs.append(str(i.message).strip())
        if not assistant_msgs:
            return ""
        return assistant_msgs[0].strip().lower()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
