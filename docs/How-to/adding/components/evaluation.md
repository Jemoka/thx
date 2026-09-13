# Adding an Evaluation

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

There are five evaluation base classes, each using a different inference strategy:

| Class | How it works | Use for |
|---|---|---|
| [`RolloutEvaluation`](#rollout) | Autoregressively generates text, then checks it against a ground truth | Open-ended generation, QA, classification via generation |
| [`EncodingEvaluation`](#encoding) | Single forward pass, argmax of logits | Token-level accuracy tasks |
| [`PerplexityEvaluation`](#perplexity) | Forward pass, computes NLL over all tokens, returns perplexity (lower is better) | Language modelling benchmarks |
| `BitsPerByteEvaluation` | Forward pass, computes bits per UTF-8 input byte (lower is better) | Byte-normalized language modelling scores |
| [`PerplexityComparisonEvaluation`](#perplexity-comparison) | NLL computed separately for each candidate continuation; lowest wins | Multiple-choice via likelihood |

All five share the same registration and wiring pattern — only the base class and the methods you implement change.

---

## `RolloutEvaluation` {#rollout}

The model generates tokens autoregressively from a prompt. You provide the prompt and expected answer; the framework handles batching, generation, and distributed aggregation.

**Required methods:** `name`, `__len__`, `get`, `clean`

```python
# theseus/evaluation/datasets/my_eval.py
from typing import Any, Tuple

from datasets import load_dataset

from theseus.data.datasets import ChatTemplate, ChatTurn
from theseus.data.tokenizer import decode_chat_template, encode_chat_template, get_tokenizer
from theseus.evaluation.base import RolloutEvaluation
from theseus.registry import evaluation


@evaluation("my_eval")
class MyEval(RolloutEvaluation):

    def __init__(self) -> None:
        self.ds = load_dataset("org/my-dataset", split="test")
        self.encoder = get_tokenizer()

    @property
    def name(self) -> str:
        return "my_eval"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> Tuple[str, str]:
        """Return (prompt_string, expected_answer_string)."""
        item = self.ds[indx]
        prompt = encode_chat_template(
            [ChatTurn(role="user", message=item["question"])],
            self.encoder,
            prompt=True,
            tokenize=False,
        )
        return prompt, item["answer"]

    def clean(self, y_hat: str) -> str:
        """Extract the model's answer from its full generation."""
        chats: ChatTemplate = decode_chat_template(y_hat)
        for turn in chats:
            if turn.role == "assistant":
                return turn.message.strip()
        return ""

    def check(self, y: str, y_hat: str) -> bool:
        return y.strip().lower() == y_hat.strip().lower()
```

**Optional overrides:**

`check(y, y_hat) -> bool` — how to compare cleaned output to expected. Default raises `NotImplementedError`, so you must override either this or `score`.

```python
def check(self, y: str, y_hat: str) -> bool:
    return y.strip().lower() == y_hat.strip().lower()
```

`score(ys, y_hats) -> list[float]` — return one score per example. The framework applies the requested mean, sum, or no reduction:

```python
def score(self, ys: list[str], y_hats: list[str]) -> list[float]:
    return [float(y in y_hat) for y, y_hat in zip(ys, y_hats)]
```

`max_new_tokens(inference) -> int` — how many tokens to generate. Defaults to `block_size`. Most tasks only need 10–256:

```python
def max_new_tokens(self, inference: Any) -> int:
    return 32
```

---

## `EncodingEvaluation` {#encoding}

No generation — a single forward pass is run and the argmax of the logit at each position is taken as the model's prediction. Good for tasks where the answer is a single next token.

**Required methods:** `name`, `__len__`, `get`, `clean`

`get` returns only the **input string** (no expected answer separately — the answer is implicit in the next token of the input).

```python
from datasets import load_dataset
from theseus.evaluation.base import EncodingEvaluation
from theseus.registry import evaluation


@evaluation("my_encoding_eval")
class MyEncodingEval(EncodingEvaluation):

    def __init__(self) -> None:
        self.ds = load_dataset("org/my-dataset", split="test")

    @property
    def name(self) -> str:
        return "my_encoding_eval"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        """Return the full input string (including the target token at the end)."""
        return self.ds[indx]["text"]

    def clean(self, y_hat: str) -> str:
        """Normalise the decoded argmax prediction."""
        return y_hat.strip()
```

`check(x, y_hat) -> bool` receives the original input string and the decoded argmax — override it to define what "correct" means:

```python
def check(self, x: str, y_hat: str) -> bool:
    # e.g. check whether the predicted last token matches what we expect
    expected_last_word = x.split()[-1]
    return expected_last_word in y_hat
```

---

## `PerplexityEvaluation` {#perplexity}

Runs a forward pass over the dataset and computes mean NLL across all non-padding tokens. Returns perplexity, so lower is better. The aggregate uses total token NLL divided by total scored tokens before exponentiation. No `clean` or `check` needed — scoring is entirely automatic.

**Required methods:** `name`, `__len__`, `get`

```python
from datasets import load_dataset
from theseus.evaluation.base import PerplexityEvaluation
from theseus.registry import evaluation


@evaluation("my_ppl_eval")
class MyPplEval(PerplexityEvaluation):

    def __init__(self) -> None:
        self.ds = load_dataset("org/my-corpus", split="test")

    @property
    def name(self) -> str:
        return "my_ppl_eval"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> str:
        """Return the text to compute perplexity over."""
        return self.ds[indx]["text"]
```

Each document is truncated to `block_size` before scoring. To report bits per
byte instead, inherit `BitsPerByteEvaluation` from the same module and implement
the same `name`, `__len__`, and `get` interface. That metric is also lower-is-better;
it normalizes NLL by the UTF-8 bytes of the evaluation input texts.

---

## `PerplexityComparisonEvaluation` {#perplexity-comparison}

Multiple-choice via likelihood: for each question the model scores every candidate continuation by its NLL (on the continuation tokens only, not the shared prefix). The candidate with the lowest NLL is the model's answer.

**Required methods:** `name`, `__len__`, `get`

`get` returns a `(prefix, continuations, correct_index)` triple:

```python
from typing import Tuple
from datasets import load_dataset
from theseus.evaluation.base import PerplexityComparisonEvaluation
from theseus.registry import evaluation


@evaluation("my_mc_eval")
class MyMCEval(PerplexityComparisonEvaluation):

    def __init__(self) -> None:
        self.ds = load_dataset("org/my-mc-dataset", split="test")

    @property
    def name(self) -> str:
        return "my_mc_eval"

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, indx: int) -> Tuple[str, list[str], int]:
        """Return (shared_prefix, list_of_continuations, correct_index)."""
        item = self.ds[indx]
        prefix = f"Question: {item['question']}\nAnswer:"
        choices = item["choices"]          # e.g. ["Paris", "London", "Berlin", "Rome"]
        correct = item["answer_index"]     # e.g. 0
        return prefix, choices, correct
```

The framework concatenates `prefix + continuation` for each choice, runs a forward pass on all of them, masks out the prefix tokens so only the continuation NLL counts, and picks the choice with the minimum mean NLL.

---

## Registering and wiring in

All five types register the same way:

```python
# theseus/evaluation/datasets/__init__.py  — add one line
from .my_eval import MyEval  # noqa: F401
```

That import is for an in-repository module. For an external module, import it
in your notebook or use the CLI's global `--import ./my_eval.py` option. Decorators
run when the module is imported; arbitrary files are not scanned.

Then declare evaluation classes on the trainer or evaluator:

```python
class MyTrainer(BaseTrainer):
    # Also declare MODEL, CONFIG, DATASET, and other experiment components.
    EVALUATION = [MyEval, MyPplEval]
```

Import `BaseTrainer` from `theseus.training.base` and the evaluation classes from
their definition modules. `eval.evaluations` is no longer a YAML selection API.
The declared classes contribute their configuration schemas automatically.

Scores are logged under evaluation names in the node store and mirrored to Comet
when enabled. A standalone `Evaluator.run()` also writes `results.json` through
`spec.result()`, beneath the cluster's results directory and project/group/run.

See [Evaluation System](../../evaluation.md) for complete training and standalone
checkpoint workflows, example limits, and output locations.
