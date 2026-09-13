from dataclasses import dataclass
import random
from collections.abc import Sequence

from theseus.data.datasets import StringDataset
from theseus.registry import dataset
from theseus.config import configure, field


@dataclass(kw_only=True)
class AdditionConfig:
    """Deterministic task parameters, owned by this dataset family."""

    samples: dict[str, int] = field("data/addition/samples")
    seeds: dict[str, int] = field("data/addition/seeds")
    left_min: int = field("data/addition/left_min")
    left_max: int = field("data/addition/left_max")
    right_min: int = field("data/addition/right_min")
    right_max: int = field("data/addition/right_max")
    operand_width: int = field("data/addition/operand_width")
    answer_width: int = field("data/addition/answer_width")
    tokens: dict[str, int] = field("data/addition/tokens")
    split: str = field("data/addition/split", default="train")


@dataset("addition")
class Addition(StringDataset):
    """Fixed-width addition serialized for ``TrivialTokenizer``.

    The config owns the IDs for ``<eos>``, ``<bos>``, ``<mid>``, ``+``, ``=``,
    ``|``, and the first digit. With IDs 0..5 and digit offset 6, the example
    ``237 + 682 = 0919`` is presented to the model as::

        1 8 9 13 3 12 14 8 4 2 6 15 7 15 0

    ``deserialize`` renders that same sequence for result artifacts as::

        <bos> 2 3 7 + 6 8 2 = <mid> 0 9 1 9 <eos>

    Thus ``<mid>`` ends the evaluation prompt, the four digit IDs are the
    expected generation, and ``<eos>`` terminates it.
    """

    CONFIG = AdditionConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        try:
            self.length = int(args.samples[args.split])
            self.left_min = int(args.left_min)
            self.left_max = int(args.left_max)
            self.right_min = int(args.right_min)
            self.right_max = int(args.right_max)
            self.operand_width = int(args.operand_width)
            self.answer_width = int(args.answer_width)
            self.eos_token = int(args.tokens["eos"])
            self.bos_token = int(args.tokens["bos"])
            self.middle_token = int(args.tokens["middle"])
            self.plus_token = int(args.tokens["plus"])
            self.equals_token = int(args.tokens["equals"])
            self.separator_token = int(args.tokens["separator"])
            self.digit_offset = int(args.tokens["digit_offset"])
        except KeyError as error:
            raise ValueError(f"addition config is missing {error.args[0]!r}") from error

        if self.length < 1:
            raise ValueError("addition samples must be positive")
        if self.left_min > self.left_max or self.right_min > self.right_max:
            raise ValueError("addition operand minima cannot exceed maxima")
        if self.operand_width < 1 or self.answer_width < 1:
            raise ValueError("addition widths must be positive")
        if len(str(max(self.left_max, self.right_max))) > self.operand_width:
            raise ValueError(
                "addition operand_width cannot represent the configured range"
            )
        if len(str(self.left_max + self.right_max)) > self.answer_width:
            raise ValueError("addition answer_width cannot represent the largest sum")

        self.token_labels = {
            self.eos_token: "<eos>",
            self.bos_token: "<bos>",
            self.middle_token: "<mid>",
            self.plus_token: "+",
            self.equals_token: "=",
            self.separator_token: "|",
            **{self.digit_offset + digit: str(digit) for digit in range(10)},
        }
        if self.eos_token != 0:
            raise ValueError("addition eos token must match TrivialTokenizer token 0")
        if set(self.token_labels) != set(range(16)):
            raise ValueError("addition token IDs must uniquely cover 0 through 15")

        # One deterministic sequence of distinct problems; configured splits
        # are slices of it. The standard tokenizer splits training rows into val.
        if any(count < 0 for count in args.samples.values()) or sum(
            args.samples.values()
        ) > (self.left_max - self.left_min + 1) * (self.right_max - self.right_min + 1):
            raise ValueError("Requested samples exceed the distinct operand tuples")
        offset = sum(count for name, count in args.samples.items() if name < args.split)
        rng = random.Random(str(sorted(args.seeds.items())))
        ranges = [(self.left_min, self.left_max), (self.right_min, self.right_max)]
        rows: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while len(rows) < offset + self.length:
            values = tuple(rng.randint(lo, hi) for lo, hi in ranges)
            if values not in seen:
                seen.add(values)
                rows.append(values)
        self.rows = rows[offset:]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> str:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        left = f"{self.rows[idx][0]:0{self.operand_width}d}"
        right = f"{self.rows[idx][1]:0{self.operand_width}d}"
        total = f"{int(left) + int(right):0{self.answer_width}d}"
        token_ids = [
            self.bos_token,
            *(self.digit_offset + int(digit) for digit in left),
            self.plus_token,
            *(self.digit_offset + int(digit) for digit in right),
            self.equals_token,
            self.middle_token,
            *(self.digit_offset + int(digit) for digit in total),
            self.eos_token,
        ]
        return " ".join(str(token_id) for token_id in token_ids)

    def deserialize(self, token_ids: Sequence[int]) -> str:
        """Render model token IDs as the task symbols used in result artifacts."""
        try:
            return " ".join(self.token_labels[int(token_id)] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown addition token ID {error.args[0]}") from error
