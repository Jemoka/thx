"""
Perplexity evaluations for existing datasets with validation splits.

These complement the existing RolloutEvaluation counterparts by measuring
how well the model predicts validation-set tokens (1/perplexity, higher is
better), which is especially useful for tracking forgetting in continual
learning experiments.
"""

from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import encode_chat_template, get_tokenizer
from theseus.data.datasets import ChatTurn


@dataclass
class MNLIPerplexityEvalConfig:
    num_samples: int = field("eval/mnli_ppl/num_samples", default=500)


@evaluation("mnli_ppl")
class MNLIPerplexityEval(PerplexityEvaluation):
    """Perplexity on MNLI validation_matched split."""

    CONFIG = MNLIPerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("nyu-mll/multi_nli", split="validation_matched")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "mnli_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
        chat = [
            ChatTurn(
                role="user",
                message=(
                    "Does the hypothesis entail the premise? Please respond "
                    'only with "entailment", "contradiction", or "neutral", '
                    "not including quotes.\n\n"
                    f"premise: {item['premise']}\n"
                    f"hypothesis: {item['hypothesis']}\n"
                ),
            ),
            ChatTurn(role="assistant", message=labels.get(item["label"], "neutral")),
        ]
        return encode_chat_template(chat, self.encoder, tokenize=False)


@dataclass
class QQPPerplexityEvalConfig:
    num_samples: int = field("eval/qqp_ppl/num_samples", default=500)


@evaluation("qqp_ppl")
class QQPPerplexityEval(PerplexityEvaluation):
    """Perplexity on QQP validation split."""

    CONFIG = QQPPerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "qqp_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        label = "yes" if item["label"] == 1 else "no"
        chat = [
            ChatTurn(
                role="user",
                message=(
                    "Are the following two questions paraphrases of each other? "
                    'Respond with "yes" or "no", not including quotes.\n\n'
                    f"question1: {item['question1']}\n"
                    f"question2: {item['question2']}\n"
                ),
            ),
            ChatTurn(role="assistant", message=label),
        ]
        return encode_chat_template(chat, self.encoder, tokenize=False)


@dataclass
class SST2PerplexityEvalConfig:
    num_samples: int = field("eval/sst2_ppl/num_samples", default=500)


@evaluation("sst2_ppl")
class SST2PerplexityEval(PerplexityEvaluation):
    """Perplexity on SST-2 validation split."""

    CONFIG = SST2PerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("stanfordnlp/sst2", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "sst2_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        label = "positive" if item["label"] == 1 else "negative"
        chat = [
            ChatTurn(
                role="user",
                message=(
                    "Classify the sentiment of the following sentence; respond "
                    'with "positive" or "negative", not including quotes.\n\n'
                    f"sentence: {item['sentence']}\n"
                ),
            ),
            ChatTurn(role="assistant", message=label),
        ]
        return encode_chat_template(chat, self.encoder, tokenize=False)


@dataclass
class SIQAPerplexityEvalConfig:
    num_samples: int = field("eval/siqa_ppl/num_samples", default=500)


@evaluation("siqa_ppl")
class SIQAPerplexityEval(PerplexityEvaluation):
    """Perplexity on Social IQa validation split."""

    CONFIG = SIQAPerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("lighteval/siqa", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "siqa_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        answers = {1: "A", 2: "B", 3: "C"}
        chat = [
            ChatTurn(
                role="user",
                message=(
                    f"Context: {item['context']}\n"
                    f"Question: {item['question']}\n\n"
                    f"A: {item['answerA']}\n"
                    f"B: {item['answerB']}\n"
                    f"C: {item['answerC']}\n\n"
                    "Answer with only the letter (A, B, or C):"
                ),
            ),
            ChatTurn(
                role="assistant",
                message=answers.get(int(item["label"]), "A"),
            ),
        ]
        return encode_chat_template(chat, self.encoder, tokenize=False)


@dataclass
class WinograndePerplexityEvalConfig:
    num_samples: int = field("eval/winogrande_ppl/num_samples", default=500)


@evaluation("winogrande_ppl")
class WinograndePerplexityEval(PerplexityEvaluation):
    """Perplexity on Winogrande validation split."""

    CONFIG = WinograndePerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        self.ds = ds.select(range(min(num_samples, len(ds))))
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "winogrande_ppl"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        item = self.ds[indx]
        answer = "A" if item["answer"] == "1" else "B"
        chat = [
            ChatTurn(
                role="user",
                message=(
                    'Fill in the blank (represented by "_") in the sentence. '
                    'Answer in a single letter, "A" or "B", without quotes.\n\n'
                    f"sentence: {item['sentence']}\n\n"
                    f"A: {item['option1']}\n"
                    f"B: {item['option2']}\n"
                ),
            ),
            ChatTurn(role="assistant", message=answer),
        ]
        return encode_chat_template(chat, self.encoder, tokenize=False)


@dataclass
class FineWebPerplexityEvalConfig:
    num_samples: int = field("eval/fineweb_ppl/num_samples", default=500)


@evaluation("fineweb_ppl")
class FineWebPerplexityEval(PerplexityEvaluation):
    """Perplexity on a sample from FineWeb."""

    CONFIG = FineWebPerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            "CC-MAIN-2022-21",
            split="train",
            streaming=True,
        )
        self.items: list[str] = []
        for item in ds:
            text = item.get("text", "")
            if text:
                self.items.append(text)
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "fineweb_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]


@dataclass
class PileDetoxifyPerplexityEvalConfig:
    num_samples: int = field("eval/pile_detoxify_ppl/num_samples", default=500)


@evaluation("pile_detoxify_ppl")
class PileDetoxifyPerplexityEval(PerplexityEvaluation):
    """Perplexity on a sample from tomekkorbak/pile-detoxify."""

    CONFIG = PileDetoxifyPerplexityEvalConfig

    def __init__(self) -> None:
        num_samples = configure(self.CONFIG).num_samples
        ds = load_dataset(
            "tomekkorbak/pile-detoxify",
            split="train",
            streaming=True,
        )
        self.items: list[str] = []
        for item in ds:
            texts = item.get("texts", [])
            if texts:
                self.items.append(" ".join(texts))
            if len(self.items) >= num_samples:
                break

    @property
    def name(self) -> str:
        return "pile_detoxify_ppl"

    def __len__(self) -> int:
        return len(self.items)

    def get(self, indx: int) -> str:
        return self.items[indx]
