import lzma
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from theseus.config import configure, field
from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


_BASE_URL = "https://data.statmt.org/cc-aligned/sentence-aligned/"


@dataclass
class CCAlignedConfig:
    language: str = field("data/ccaligned/language", default="fr_XX")


@dataset("ccaligned")
class CCAligned(StreamingPretrainingDataset):
    """Multilingual text from CCAligned.

    Streams target-language sentences directly from statmt.org,
    decompressing on-the-fly without downloading the whole file first.
    The ``config`` parameter selects the target language code
    (e.g. ``"fr_XX"``, ``"de_DE"``, ``"zh_CN"``).  Each yielded string
    is a single target-language sentence, suitable for monolingual
    pretraining in that language.

    Some language pairs are stored as ``en_XX-{lang}.tsv.xz`` and others
    as ``{lang}-en_XX.tsv.xz``; both orderings are tried automatically.
    """

    CONFIG = CCAlignedConfig

    def __init__(self) -> None:
        self.lang = configure(self.CONFIG).language
        self._urls = [
            f"{_BASE_URL}en_XX-{self.lang}.tsv.xz",
            f"{_BASE_URL}{self.lang}-en_XX.tsv.xz",
        ]

    def _open(self) -> Any:
        for url in self._urls:
            try:
                return urllib.request.urlopen(url)  # noqa: S310
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
        raise FileNotFoundError(
            f"CCAligned data not found for {self.lang}; tried {self._urls}"
        )

    def __iter__(self) -> Iterator[str]:
        response = self._open()
        decompressor = lzma.LZMADecompressor()
        leftover = b""
        for chunk in iter(lambda: response.read(64 * 1024), b""):
            raw = decompressor.decompress(chunk)
            lines = (leftover + raw).split(b"\n")
            leftover = lines.pop()
            for line in lines:
                parts = line.decode("utf-8", errors="replace").split("\t")
                if len(parts) >= 2:
                    tgt_text = parts[1].strip()
                    if tgt_text:
                        yield tgt_text
        # Handle final leftover
        if leftover:
            parts = leftover.decode("utf-8", errors="replace").split("\t")
            if len(parts) >= 2:
                tgt_text = parts[1].strip()
                if tgt_text:
                    yield tgt_text
        response.close()
