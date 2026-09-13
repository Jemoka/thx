"""
fever dataset
Though! We use the Wikipedia API so we don't have to download the entire dump they have. But lowk
that maybe the best idea so TBD.
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
DEV_URL = "https://fever.ai/download/fever/paper_dev.jsonl"


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


def template(claim: str, evidence_text: str | None, label: str) -> ChatTemplate:
    if evidence_text:
        message = f"""Given the following claim and evidence, determine whether the claim is supported by the evidence, refuted by the evidence, or if there is not enough information to verify it.

Respond with exactly one of: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"

Claim: {claim}

Evidence: {evidence_text}"""
    else:
        message = f"""Given the following claim, determine whether it is supported by evidence, refuted by evidence, or if there is not enough information to verify it.

Respond with exactly one of: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"

Claim: {claim}"""

    return [
        ChatTurn(role="user", message=message),
        ChatTurn(role="assistant", message=label),
    ]


@dataclass
class FEVERConfig:
    split: str = field("data/fever/split", default="train")


@dataset("fever")
class FEVER(ChatTemplateDataset):
    CONFIG = FEVERConfig

    def __init__(self) -> None:
        split = configure(self.CONFIG).split
        if split == "train":
            url = TRAIN_URL
        elif split in ("dev", "validation"):
            url = DEV_URL
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'dev'")

        self.data: list[dict[str, Any]] = []
        with urlopen(url) as resp:
            for line in resp:
                self.data.append(json.loads(line))

        self._wiki_cache: dict[str, str | None] = {}

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.data[idx]
        claim = item["claim"]
        label = item["label"]
        evidence_text = self._get_evidence_text(item["evidence"])
        return template(claim, evidence_text, label)
