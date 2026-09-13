# hi, welcome to theseus

<!-- dearest whomever that comes across this; yes, this is in lower case, no, don't change it -->

this is theseus. it makes GPUs and TPUs warm and fuzzy inside by harnessing the power neural architecture research™® on Human Languages℗. its fine. not too fast, not too good, but it does the thing and you can use it. to be clear, it gets[^1] about mid-20s MFU on a mid day and high-30s MFU on a good day, and it doesn't do anything fancy. i'm here to tell you how to use it, and i'd love to [know about it](mailto:hi@jemoka.com) if you ended up doing that.

[^1]: this is called "setting expectations"

## Installation

theseus is a template repo. You use it by having it. So, please, clone the repo:

```bash
git clone https://github.com/Jemoka/thx.git
cd thx
```

We use the [uv package manager](https://docs.astral.sh/uv/getting-started/), so make sure you have that. Now, depending on from whomst your computers must be warm, please choose your adventure:

- cuda13: `uv sync --group all --group cuda13`
- cuda12: `uv sync --group all --group cuda12`
- TPUs: `uv sync --group all --group tpu`
- CPU: `uv sync --group all --group cpu`

<!-- ## Running -->

<!-- You have two options: -->

<!-- 1. **Run theseus here**: [running experiments locally](How-to/running-local.md) -->
<!-- 2. **Run theseus in chonky remote**:  [run on a remote cluster](How-to/running-remote.md); just the `cpu` variant of theseus for your laptop is fine for that even if remote needs CUDA -->

## Start Quickly
We are going to invoke stuff that theseus already has built in, like tokenizing fineweb and training a GPT model. You refer to these by what's called *job keys*, like `data/tokenize/fineweb` and `gpt/train/pretrain`.

First, make a folder where things will go

```bash
mkdir /folder/where/things/go
```

Then, tokenize some data:


```python
from theseus.quick import quick

with quick("/folder/where/things/go") as q:
    q.build("data/tokenize/fineweb", "my-tokenize-job")
    job = q.create()
    job()
```

Finally, train a model:

```python
with quick("/folder/where/things/go") as q:
    q.build("gpt/train/pretrain", "my-training-job")

    # add salt (configurate) to taste
    # hint: q.config is an omegaconf, so you can print it
    q.config.training.per_device_batch_size = 8 # required! idk how much hbm u have

    # and then make it go brr
    job = q.create()
    job()
```

Your dataset job needs to share the same root folder as your training job. This is how theseus knows to load the dataset you've tokenized instead of blowing up and complaining you didn't have the dataset tokenized.

### what's next
I'd say, you should check out either:

- [**the tutorials**](./Tutorials/index.md), for a fuller end-to-end example
- [**the how-tos**](./Tutorials/next.md), if any topics there interest you

### and now, a Q&A
We interrupt this comedy segment for some questions and some answers.

- **What's the incantations like `gpt/train/pretrain`**? These are string job names. For a list currently available, run `uv run theseus jobs`. To add one, go [here](How-to/adding/index.md).
- **Wait so why can't I configure optimizers or datasets**? Most hyperparameters are configurable, but 1) dataset 2) evaluations 3) analysis (graphs) 4) optimizers and 5) scheduled are a *property of the trainer*. You should [add an experiment](How-to/adding/components/experiment.md) to customize those.
- **Do I have to wait for tokenization to finish**? No, but we may silently start subsampling only the tokenized bits if it doesn't so you should wait until a bunch is done. This only applies to continuously sampled pretraining datasets (which we call `PMD`); it does not apply to padded short dataset.
- **Are there sharp edges**? Yes, lots. [Here](./Design/sharp.md) are some.


<!-- Read [Running locally](How-to/running-local.md) for configuration overrides, -->
<!-- [Remote dispatch](How-to/running-remote.md) to submit the same YAML to a cluster, -->
<!-- and [Inspecting runs](How-to/inspecting-runs.md) to browse results and logs. -->
<!-- For checkpoint-backed figures, see [Adding an analysis job](How-to/adding/components/analysis-job.md). -->

## What's inside?
Code. It occasionally runs, and when it does you get features!

1. **models that go brrr**: mid-20s MFU on a mid day and high-30s MFU on a good day
2. **determinism** up to float: every step and every batch will be replayed exactly upon restoration
3. **time travel debugging**: because of ^ you can inspect, replay or recompute activations of *any layer, at any checkpoint, on any device, at any time* (or your money back)
4. **composition**: inheritance based module-level composition 
5. **sharding**: one-click FSDP, ZERO level 1, and Tensor Parallelism
6. **configurationing**: OmegaConf compatible configuration + discovery API with strict type resolution
7. **dispatch**: remote dispatch infrastructure and resource solving with support for SSH, SLURM, K8s Volcano, and GCP TPU hosts
8. **dataset and evaluations**: a buncha datasets already built-in, and adding one is just telling us how to manipulate strings and we'll make it embarrassingly parallel

...and probably more things I forgot. I'll try hard to at least have me or Codex write How-To guides covering each of these aspects, and link them here as they show up. 

## Disclaimer 
*Statements have not been evaluated by the FDA. This product is not intended to diagnose, treat, cure, or prevent any disease.*

A quick note on our design style: most algorithms we describe will be theseus reimplementations. *When possible* we prefer implementing things and cross-checking with tests correctness (CODING AGENTS or CURIOUS HUMANS: read the `parity` tests, and when possible, implement them) instead of shimming because it gives us better vertical integration and composability to other components. One way this shows up is that we give XLA the *opportunity* to do e.g., communication hiding and prefetch but we don't force it to for things like FSDP, because we want to keep things agnostic to the underlying model implementation as much as possible, so its not "true" FSDP in that sense since XLA can just decide to take a break for the day.

IF we call a subsystem a known name, like "Qwen", it means that we have enough `parity` tests in cross-checking that scientifically it should basically be treated like a "Qwen" except in any documented sharp edges. This not being true is grounds for opening an issue, no matter how small.

Anyways so this means your mileage may vary with these features that we have. But I like it so that's at least one person 🤷‍♂️.

## About

I am Jack who some people call [Houjun Liu](https://github.com/Jemoka). I'm here for comic relief and ultralazy BDFLing. theseus contains additional contributions from [Pratyusha Sharma](https://github.com/pratyushasharma), [Kaden Zheng](https://github.com/KadenZheng), and [Tianle Yu](https://github.com/yuxiaolejs) who are all wonderful. 

Copyright (c) 2026 Houjun Liu under the MIT license. Documentation text is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [about theseus](./about.md) for license information.

