# Adding a Dataset

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

A dataset yields text for pretraining or chat turns for fine-tuning. There are four types of datasets, each of which you can inherit per your needs.

## Dataset types

| Base class | Data shape | When to use |
|---|---|---|
| `StreamingPretrainingDataset` | yields `str` | Large pretraining corpora; tokenised irrespective of item boundaries |
| `PretrainingDataset` | `__getitem__` → `str` | Same semantics but finite and indexable |
| `ChatTemplateDataset` | `__getitem__` → `ChatTemplate` | Instruction / chat fine-tuning |
| `StreamingChatTemplateDataset` | yields `ChatTemplate` | Streaming chat data |

---

## Pretraining dataset (streaming)

This is the most common case for large-scale pretraining data from HuggingFace:

```python
# theseus/data/datasets/my_corpus.py
from collections.abc import Iterator

from datasets import load_dataset
from theseus.data.datasets import StreamingPretrainingDataset
from theseus.registry import dataset


@dataset("my_corpus")
class MyCorpus(StreamingPretrainingDataset):
    def __init__(self, config: str | None = None) -> None:
        self.ds = load_dataset("org/my-corpus", split="train", streaming=True)

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            yield item["text"]
```

Register it:

```python
# theseus/data/datasets/__init__.py  — add one line
from .my_corpus import MyCorpus  # noqa: F401
```

---

## Chat / instruction dataset

For fine-tuning on instruction-following data, yield `ChatTemplate` — a list of `ChatTurn` objects:

```python
# theseus/data/datasets/my_chat.py
from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


@dataset("my_chat")
class MyChat(ChatTemplateDataset):
    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("org/my-chat-dataset", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        return [
            ChatTurn(role="user",      message=item["instruction"]),
            ChatTurn(role="assistant", message=item["response"]),
        ]
```

A `ChatTemplate` is just a `list[ChatTurn]`. Roles are free-form strings but `"user"`, `"assistant"`, and `"system"` are the conventional values.

---

For an external dataset, import its module in the notebook or load it with
`uv run theseus --import ./my_dataset.py ...`. Editing the built-in package is
only needed when adding the implementation to that package itself.

## Dataset with multiple splits

A `split` constructor argument supports direct Python callers. Tokenization jobs
instantiate `DATASET()` without arguments, so expose configurable source options
through a dataset `CONFIG` dataclass when they must be editable in YAML:

```python
@dataset("my_corpus")
class MyCorpus(StreamingPretrainingDataset):
    def __init__(self, split: str = "train", config: str | None = None) -> None:
        self.ds = load_dataset("org/my-corpus", split=split, streaming=True)

    def __iter__(self) -> Iterator[str]:
        for item in self.ds:
            yield item["text"]
```

---

## Using your dataset in an experiment

Declare the dataset on the trainer class with `DATASET`. The training loader
reads materialized data, so tokenize the corpus before training. Dataset choices
and sampling rates are Python declarations, not `training.dataset` YAML fields:

```python
from theseus.data.datasets import FineWeb
from theseus.training.flywheel.strategy import Sampling

class MyTrainer(BaseTrainer):
    # Also declare MODEL, CONFIG, and the other experiment components.
    DATASET = [
        Sampling(FineWeb, 0.8, "pmd"),
        Sampling(MyCorpus, 0.2, "pmd"),
    ]
```

Import `BaseTrainer` from `theseus.training.base` and `MyCorpus` from your dataset
module. Rates must sum to one. Use `"padded"` for materialized chat data and
`"pmd"` for contiguous pretraining tokens. Source options belong to a dataset's
`CONFIG` schema; see [Adding an experiment](experiment.md).

## Materialize before training

Declare `DATASET = MyCorpus` on a `TokenizeVariableDatasetJob` subclass for
contiguous pretraining data. For padded chat examples, use
`TokenizeBlockwiseDatasetJob`. Both classes live in `theseus.data.tokenize`.
Register the preparation job with `@job`, build it through `quick`, and call the
created job. Use the same tokenizer, dataset key, and `data.suffix` for the
training reader. 
