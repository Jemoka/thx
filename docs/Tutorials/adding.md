# Making It Your Own

You've got a trained GPT on Fineweb, plaudits to all involved! But there wasn't an awful a lot of control you've had in what is actually being trained. This is sad, so let's make it less sad. Our goal now will be to tie the lessons from the previous few steps together but to train a model you actually control on a dataset you actually made.

## Making a dataset
There's actually four dataset types in theseus. But we are currently not going to care about this beyond knowing that `StreamingPretrainingDataset` is, heh, a streaming pretraining dataset which will get tokenized via standard *document packing* (but, importantly, without attention masks). Let's make one of them now:

```python
from dataclasses import dataclass
from datasets import load_dataset

# configuration system
from theseus.config import configure, field
# discovery system
from theseus.registry import dataset
# dataset type
from theseus.data.datasets import StreamingPretrainingDataset

@dataclass
class MyFineWebConfig:
    chunk: str = field("data/my_fineweb/snapshot", default="CC-MAIN-2022-21")

@dataset("my-fineweb")
class MyFineWeb(StreamingPretrainingDataset):
    CONFIG = MyFineWebConfig

    def __init__(self):
        # hydrate configuration
        args = configure(self.CONFIG)
        
        # get a huggingface dataset handle
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb",
            args.chunk,
            split="train",
            streaming=True,
        )

    def __iter__(self):
        # generate the dataset as an iterator
        for i in self.ds["text"]:
            yield i
```

Woah woah. What's going on. This is actually a great moment to pause and talk a bit about another new new theseus idea, which is the **configuration system**. It contains three components:

- `CONFIG` class property: datasets and evaluations have this to tell the system what fields are available by passing it a `dataclass`
- `field()` declarator: inside every configuration, we use the `field` decorator to declare what fields are available; it asks for a `/` delineated scoped configuration field
- `configure(...)` construct a instance of the passed-in configuration dataclass by resolving each `field` from the job's declared configuration, like when you pass it in by `QuickJob.config`

Remember how in the previous chapter our `QuickJob` object had `q.config`? It came from these fields. The only thing you'll need to know about these fields is that fields with the same name will become the same keys. For instance, if two subsystems request `field("architecture/block_size")`, `q.config.architecture.block_size` will be mapped to both of those classes.

## Making a tokenizer job
Now that you have a dataset, we'll have to tokenize it first to use it! This requires creating a `job` which actually does the tokenization. Theseus has two tokenizer jobs, and the one that works with streaming pretraining datasets is `TokenizeVariableDatasetJob`.

```python
from theseus.data.tokenize import TokenizeVariableDatasetJob

@job("tutorial/data/tokenize-my-fineweb")
class TokenizeLocalText(TokenizeVariableDatasetJob):
    DATASET = MyFineWeb
```

In general, theseus expresses configuration in two levels: the `field`/`configure` system, described above, and typed dependencies which you manually pass in to class properties like we see here in `DATASET`. In fact, you have seen this before: `CONFIG` is another common field that theseus components would use.

Once both of these things is in your script, I'm sure you know what to do next :) Let's fire it off:


```python
from theseus.quick import quick

with quick("/Users/houjun/theseus") as q:
    q.build(TokenizeLocalText, "my-tokenization-run")
    q.config.data.max_tokens = 1_000_000
    q.config.data.val_pct = 0.0 # we live dangerously
    job = q.create()
    job()
```

Incidentally, I also want to point out a general pattern that's visible here. Every custom object has two layers in `theseus`: WHAT you are doing (`Dataset`) and HOW you are doing it (`TokenizeVariableDatasetJob`). The WHAT is situation is dependent, and the HOW is always a registered job.

???+ note "How to remotely dispatch this?"
    It should just work™ if you stick this file into the `theseus` tree, say, under `theseus/data/datasets` and import it in `theseus/data/datasets/__init__.py`, the system will pick it up because of the `@job` and `@dataset` decorators will register these jobs upon import. If you have a script that isn't in the theseus tree, fear not! `--import` is a CLI flag that will run local imports and pack it for remote jobs. The catch is that this is a `cloudpickle`, meaning you may get hilariously giant and platform dependent traces. Even that should still just work™, but if it doesn't, open an issue please!
    
    For instance:
    
    ```bash
    uv run theseus --import my_job_script.py submit ...
    ```

## Extending a model
At this point the tutorial is basically rinse and repeat, with a small caveat. I'm going to show you how to extend a GPT model to do something custom, and you will applaud and say "great this guy knows his own library's extension points and it works almost the same as writing a dataset yayyy!" and then cry because you on the other hand don't want to spend time reading what the correct semantics for what to extend.

In the spirit of making you Way Less Sad℗ we will document common extension points in [this how-to guide](../How-to/adding/index.md). For now, we will assume that I have omniscient knowledge of the extension points and you can play along.

Let's say you want to make your logits sharper for some reasons since you are a glutton for confidence:

```python
from theseus.config import field
from theseus.model.models import GPT
from theseus.registry import job
from theseus.training.base import BaseTrainer, BaseTrainerConfig
from theseus.data.datasets import FineWeb

# make the model
class ScaledGPT(GPT):
    logit_scale: float = field("architecture/logit_scale", default=1.0)

    def unembed(self, x: jax.Array):
        return super().unembed(x) * self.logit_scale

# make the trainer
@job("scaledgpt/train/pretrain")
class ScaledGPTTrainer(BaseTrainer[BaseTrainerConfig, GPT]):
    MODEL = ScaledGPT
    CONFIG = BaseTrainerConfig
    DATASET = [Sampling(MyFineWeb, 0.5, "pmd"), Sampling(FineWeb, 0.5, "pmd")] 
    #         ^ "use MyFineWeb, 50% of the time, and its a streaming dataset"
```

Trainer classes have seven static fields for you to configurate, I showed you three. I'll let the [extension guide](../How-to/adding/index.md) handle the rest. Just for completeness let's show you how to train one of these. I think you know the drill:

```python
with quick("/Users/houjun/theseus") as q:
    q.build(ScaledGPTTrainer, "train-scaled-gpt")

    q.config.architecture.n_embd = 128
    q.config.architecture.n_layers = 4
    q.config.architecture.block_size = 128
    q.config.architecture.logit_scale = 0.8

    q.config.training.batch_size = 8
    q.config.training.per_device_batch_size = 4
    q.config.training.tokens = 32_768
    q.config.training.validation = False
    q.config.logging.report_interval = 1
    q.config.logging.checkpoint_interval = 8

    trainer = q.create()
    trainer()
```

Cool.

-----

That's the end of the tutorial. The [what's next](./next.md) lists a bunch of things that's next. So go there next. Thanks for flying Tutorial.
