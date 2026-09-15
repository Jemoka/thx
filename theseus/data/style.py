"""Reader styles for materialized datasets."""

from enum import Enum
from typing import Literal


class DatasetStyle(Enum):
    PADDED = "padded"
    PMD = "pmd"
    CONTRASTIVE = "contrastive"

    @classmethod
    def normalize(cls, style: "DatasetStyleLike") -> "DatasetStyle":
        if isinstance(style, cls):
            return style
        style_lower = str(style).lower()
        for option in cls:
            if style_lower == option.value:
                return option
        raise ValueError(f"Unknown dataset style: {style}")


DatasetStyleLike = DatasetStyle | Literal["padded", "pmd", "contrastive"]
