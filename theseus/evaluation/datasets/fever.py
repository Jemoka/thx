import json
from dataclasses import dataclass
from typing import Any, Tuple
from urllib.request import urlopen

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation
from theseus.data.tokenizer import (
    decode_chat_template,
    encode_chat_template,
    get_tokenizer,
)


DEV_URL = "https://fever.ai/download/fever/paper_dev.jsonl"


@dataclass
class FEVEREvalConfig:
    split: str = field("eval/fever/split", default="dev")


def get_wikipedia_summary(article_title: str) -> str | None:
    """Fetch Wikipedia summary for an article title like 'Oliver_Reed'."""
    try:
        import wikipedia
    except ModuleNotFoundError as exc:
        if exc.name != "wikipedia":
            raise
        raise ModuleNotFoundError(
            "FEVER evidence requires wikipedia. Install with `uv sync --group fever` "
            "or `pip install 'libthx[fever]'`."
        ) from exc

    page_name = (
        article_title.replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
    )
    try:
        result: str = wikipedia.summary(page_name, sentences=3)
        return result
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return None


def template(claim: str, evidence_text: str | None) -> ChatTemplate:
    if evidence_text:
        message = f"""Given the following claim and evidence, determine whether the claim is supported by the evidence, refuted by the evidence, or if there is not enough information to verify it.

Respond with exactly one of: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"

Claim: {claim}

Evidence: {evidence_text}"""
    else:
        message = f"""Given the following claim, determine whether it is supported by evidence, refuted by evidence, or if there is not enough information to verify it.

Respond with exactly one of: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"

Claim: {claim}"""

    return [ChatTurn(role="user", message=message)]


@evaluation("fever")
class FEVEREval(RolloutEvaluation):
    """FEVER fact verification evaluation using dev split."""

    CONFIG = FEVEREvalConfig

    def __init__(self) -> None:
        split = configure(self.CONFIG).split
        if split not in ("dev", "validation"):
            raise ValueError(f"Unknown split: {split}. Use 'dev' or 'validation'")

        self.data: list[dict[str, Any]] = []
        with urlopen(DEV_URL) as resp:
            for line in resp:
                self.data.append(json.loads(line))

        self._wiki_cache: dict[str, str | None] = {}
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "fever"

    def max_new_tokens(self, inference: Any) -> int:
        """Need ~5 tokens for SUPPORTS/REFUTES/NOT ENOUGH INFO."""
        return 25

    def _get_evidence_text(self, evidence: list[Any]) -> str | None:
        """Extract unique Wikipedia article titles and fetch summaries."""
        articles: set[str] = set()
        for evidence_group in evidence:
            for item in evidence_group:
                article_title = item[2]
                if article_title is not None:
                    articles.add(article_title)

        if not articles:
            return None

        summaries = []
        for article in articles:
            if article not in self._wiki_cache:
                self._wiki_cache[article] = get_wikipedia_summary(article)
            summary = self._wiki_cache[article]
            if summary:
                summaries.append(summary)

        return "\n\n".join(summaries) if summaries else None

    def get(self, indx: int) -> Tuple[str, str]:
        item = self.data[indx]
        claim = item["claim"]
        answer = item["label"]
        evidence_text = self._get_evidence_text(item["evidence"])
        prompt = encode_chat_template(
            template(claim, evidence_text),
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, answer

    def __len__(self) -> int:
        return len(self.data)

    def clean(self, y_hat: str) -> str:
        chats: ChatTemplate = decode_chat_template(y_hat)
        assistant_msgs = []
        for i in chats:
            if i.role == "assistant":
                assistant_msgs.append(i.message.strip())
        if not assistant_msgs:
            return ""
        return assistant_msgs[0].strip().upper()

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().upper() == y_hat.strip().upper()
