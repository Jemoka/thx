# Running Experiments

Okeeee so you [installed theseus](../quickstart.md#installation). Now let's get started. You first contact with theseus will be though the `quick` API. This API is quicker to use than the other APIs, and it actually does 80% of the things.

## make a folder
Grace your keyboard with the presence of your hands and make a folder.

```bash
mkdir /Users/houjun/theseus
```

wow, such folder. This folder will contain things™:

- datasets 
- model checkpoints
- evaluation and analysis results
- training logs

don't worry about how to find where each of these things are quite yet, we'll get that to the [next section](./inspecting-runs.md).


## build a (tokenization) job
`quick` is a way to declaratively make `job`s, which is the basic unit of cooking in theseus. But to even build a job we'll need to know what jobs are available to be built. 

Navigate to the theseus folder you cloned during installation and run:

```bash
uv run theseus jobs
```
??? example "command output"
    ```
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Job                         ┃ Description                                                            ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ data/tokenize/addition      │ Materialize one indexable dataset as padded token and mask arrays.     │
    │ data/tokenize/alpaca        │ Materialize one indexable dataset as padded token and mask arrays.     │
    │ data/tokenize/bbq           │ Materialize one indexable dataset as padded token and mask arrays.     │
    │ data/tokenize/ccaligned     │ Materialize one streaming dataset as contiguous train/validation       │
    │                             │ tokens.                                                                │
    │ data/tokenize/fineweb       │ Materialize one streaming dataset as contiguous train/validation       │
    │                             │ tokens.                                                                │
    │ data/tokenize/cfq           │ Materialize one indexable dataset as padded token and mask arrays.     │
    |     ...                     │                                ...                                     │
    └─────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

    ```

    (ok there's a lot left but copying is hard mkay)

Cute! Those are a lot of jobs! How about we start with a tokenization job. I like `data/tokenize/fineweb`. It will be our dataset for the day. Let's open this job (that is, read its details, WITHOUT running yet) with `quick` and see what we can explore.

We also have a pick a name for our instantiation of the job. Let's call it `tokenize-fineweb`. Now, open a nice Python interpreter/REPL/notebook/wicked witch, and import the quick API:

```python
from theseus.quick import quick
```

and open our job:

```python

with quick("/Users/houjun/theseus") as q:
    q.build("data/tokenize/fineweb", "tokenize-fineweb")
    print(q)
```
??? example "command output"
    ```
    <theseus.quick.QuickJob object at 0x134b39550>
    ```
    
Neat. What, pray tell, does `QuickJob` do? Well, for one, it gives you every knob you can turn about that particular job you chose. To see this, let's print the job's config.


```python
with quick("/Users/houjun/theseus") as q:
    q.build("data/tokenize/fineweb", "tokenize-fineweb")
    print(q.config)
```
??? example "command output"
    ```json
    {'data': {'val_pct': 0.05, 'seed': 2357, 'max_samples': -1, 'max_tokens': -1, 'suffix': '', 'fineweb': {'snapshot': 'CC-MAIN-2022-21'}}, 'tokenizer': {'backend': 'tiktoken', 'name': 'cl100k_base', 'huggingface': {'use_fast': True, 'use_remote_code': False}}}
    ```
    
Fun. You can even change some knobs. Since I'm feeling particularly impatient, I'm going to change the maximum number of tokens to `1e7`. 

```python
with quick("/Users/houjun/theseus") as q:
    q.build("data/tokenize/fineweb", "tokenize-fineweb")
    q.config.data.max_tokens = int(1e7)
```

And with your chakras centered let's 1) instantiate the job with our configuration and 2) kick it off tokenizing.

```python
with quick("/Users/houjun/theseus") as q:
    q.build("data/tokenize/fineweb", "tokenize-fineweb")
    q.config.data.max_tokens = int(1e7)
    
    job = q.create()
    job()
```
??? example "command output"
    ```
    2026-09-08 20:43:57.412 | DEBUG    | theseus.job:setup:218 - JOB tokenize-fineweb | building state
    2026-09-08 20:43:57.412 | DEBUG    | theseus.job:setup:240 - JOB tokenize-fineweb | starting anew
    2026-09-08 20:43:57.414 | INFO     | theseus.job:__call__:249 - JOB tokenize-fineweb | starting
    2026-09-08 20:43:57.414 | DEBUG    | theseus.job:__call__:250 - JOB tokenize-fineweb | pre-start sync
    2026-09-08 20:43:57.414 | INFO     | theseus.job:__call__:260 - JOB tokenize-fineweb | synchronized and starting with nodeid=Z2VuZXJhbC5kZWZhdWx0LnRva2VuaXplLWZpbmV3ZWI6ZDRlNTA4OjA
    2026-09-08 20:44:29.615 | INFO     | theseus.data.tokenize:run:563 - DATA | 1000 samples | 708571 train tokens | 30270 val tokens | 0.3s
    2026-09-08 20:44:29.871 | INFO     | theseus.data.tokenize:run:563 - DATA | 2000 samples | 1413636 train tokens | 68763 val tokens | 0.5s
    2026-09-08 20:44:30.117 | INFO     | theseus.data.tokenize:run:563 - DATA | 3000 samples | 2086774 train tokens | 100717 val tokens | 0.8s
    2026-09-08 20:44:30.367 | INFO     | theseus.data.tokenize:run:563 - DATA | 4000 samples | 2797780 train tokens | 134787 val tokens | 1.0s
    2026-09-08 20:44:30.598 | INFO     | theseus.data.tokenize:run:563 - DATA | 5000 samples | 3417868 train tokens | 167815 val tokens | 1.2s
    2026-09-08 20:44:30.832 | INFO     | theseus.data.tokenize:run:563 - DATA | 6000 samples | 4039742 train tokens | 199745 val tokens | 1.5s
    2026-09-08 20:44:31.089 | INFO     | theseus.data.tokenize:run:563 - DATA | 7000 samples | 4792466 train tokens | 221064 val tokens | 1.7s
    2026-09-08 20:44:31.336 | INFO     | theseus.data.tokenize:run:563 - DATA | 8000 samples | 5490534 train tokens | 253719 val tokens | 2.0s
    2026-09-08 20:44:31.582 | INFO     | theseus.data.tokenize:run:563 - DATA | 9000 samples | 6187381 train tokens | 285418 val tokens | 2.2s
    2026-09-08 20:44:31.808 | INFO     | theseus.data.tokenize:run:563 - DATA | 10000 samples | 6814021 train tokens | 311249 val tokens | 2.5s
    2026-09-08 20:44:32.038 | INFO     | theseus.data.tokenize:run:563 - DATA | 11000 samples | 7431341 train tokens | 343675 val tokens | 2.7s
    2026-09-08 20:44:32.285 | INFO     | theseus.data.tokenize:run:563 - DATA | 12000 samples | 8067113 train tokens | 369369 val tokens | 2.9s
    2026-09-08 20:44:32.527 | INFO     | theseus.data.tokenize:run:563 - DATA | 13000 samples | 8729187 train tokens | 394805 val tokens | 3.2s
    2026-09-08 20:44:32.797 | INFO     | theseus.data.tokenize:run:563 - DATA | 14000 samples | 9517975 train tokens | 421786 val tokens | 3.4s
    2026-09-08 20:44:32.820 | DEBUG    | theseus.job:__call__:267 - JOB tokenize-fineweb | finished, waiting for everyone...
    2026-09-08 20:44:32.825 | INFO     | theseus.job:__call__:272 - JOB tokenize-fineweb | finished
    ```

After basking in the godlike high of tokenizing data, realize that you still have actually train the model, so let's do that next.


## build a (training) job
Actually its the same thing the job name is `gpt/train/pretrain` this time. Let's call our job "train-gpt". Let's give it a try:

```python
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "train-gpt")
    print(q.config)
```
???+ example "command output"
    ```json
    {'architecture': {'dtype': {'param': 'float32', 'activation': 'bfloat16'}, 'n_embd': 2048, 'n_layers': 32, 'bias': True, 'dropout': 0.0, 'n_head': 16, 'block_size': 512, 'rope': True, 'vocab_size': 100288, 'instrumentation': {'residual': False, 'components': False}, 'layer_norm_eps': 1e-05, 'intermediate_size': -1}, 'eval': {'length': -1}, 'tokenizer': {'backend': 'tiktoken', 'name': 'cl100k_base', 'huggingface': {'use_fast': True, 'use_remote_code': False}}, 'data': {'suffix': '', 'fineweb': {'snapshot': 'CC-MAIN-2022-21'}}, 'optimization': {'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.95, 'lr': 0.0003, 'warmup_pct': 0.005, 'decay_pct': 0.01, 'final_lr_frac': 0.01}, 'training': {'batch_size': 512, 'per_device_batch_size': -1, 'tokens': 1000000000, 'warmup_pct': 0.01, 'decay_pct': 0.1, 'validation': True, 'evaluate': True, 'analyze': True, 'validation_steps': 2048}, 'logging': {'report_interval': 32, 'checkpoint_interval': 1024, 'validation_interval': 512, 'remote': False}}
    ```
    
Woah woah woah that's too much to scroll. You maybe delighted to know that I too love configs but hate scrolling. So, `q.config` is actually an `OmegaConf`, which is a nice configuration system. We can use this fact to our advantage by seeing a YAML view of the configuration instead by asking `OmegaConf` for it.

```python
from omegaconf import OmegaConf
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "train-gpt")
    print(OmegaConf.to_yaml(q.config))
```
??? example "command output"
    ```yaml
    architecture:
        dtype:
            param: float32
            activation: bfloat16
        n_embd: 2048
        n_layers: 32
        bias: true
        dropout: 0.0
        n_head: 16
        block_size: 512
        rope: true
        vocab_size: 100288
        instrumentation:
            residual: false
            components: false
        layer_norm_eps: 1.0e-05
        intermediate_size: -1
    eval:
        length: -1
    tokenizer:
        backend: tiktoken
        name: cl100k_base
        huggingface:
            use_fast: true
            use_remote_code: false
    data:
        suffix: ''
        fineweb:
            snapshot: CC-MAIN-2022-21
    optimization:
        weight_decay: 0.1
        beta1: 0.9
        beta2: 0.95
        lr: 0.0003
        warmup_pct: 0.005
        decay_pct: 0.01
        final_lr_frac: 0.01
    training:
        batch_size: 512
        per_device_batch_size: -1
        tokens: 1000000000
        warmup_pct: 0.01
        decay_pct: 0.1
        validation: true
        evaluate: true
        analyze: true
        validation_steps: 2048
    logging:
        report_interval: 32
        checkpoint_interval: 1024
        validation_interval: 512
        remote: true
    ```

Great, let's move... wait a minute. look at that config. wtf?

**through mind reading trick `data.fineweb` already there?** so ok that's a bit of misdirection, sorry. we tokenized `fineweb` with `cl100k` tokenizer, and it just so happens that `gpt/train/pretrain` job uses exactly that dataset. Datasets are *part of the job itself* in `theseus`. If you'd like to swap out a dataset, you should add your own training job that uses a different dataset. We'll get to this.

**why is `training.per_device_batch_size=-1`?** because I don't know how VRAM-rich you are. `-1` asks the execution bootstrap to probe progressively smaller microbatch sizes while preserving `training.batch_size` through gradient accumulation. Set `THESEUS_AUTOBATCH_SIZE_CANDIDATES` to a space-separated descending list such as `"8 4 2 1"` when prior measurements give you a tighter search range.

ok ok so let's set a configuration for the batch size. I'm also going to make my GPT model hilariously small because that's cute. you don't have to if you like it big. I'm also going to set reporting (logging) and checkpointing both to every batch. This will be useful for our analysis next chapter so we don't have to sit around waiting for checkpoints.

```python
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "train-gpt")

    q.config.training.per_device_batch_size = 4
    q.config.architecture.n_embd = 128
    q.config.architecture.n_layers = 4
    q.config.logging.report_interval = 1
    q.config.logging.checkpoint_interval = 1

    job = q.create()
    job()
```
??? example "command output"
    ```
    2026-09-08 21:02:55.532 | INFO     | theseus.training.base:__init__:187 - TOPOLOGY | 
    {
    "name": "train-gpt",
    "id": null,
    "project": null,
    "group": null,
    "execution_id": null,
    "tag": null,
    "topology": {
        "shard": {
        "tp": 1,
        "fsdp": false,
        "zero": true
        },
        "chip": {
        "name": "cpu",
        "display_name": "CPU",
        "memory": 34359738368,
        "flops": {
            "bfloat16": null,
            "float32": null
        }
        },
        "device_count": 1,
        "local_device_count": 1,
        "process_count": 1,
        "is_main": true,
        "replicas": 1,
        "local_replicas": 1
    },
    "hardware": {
        "chip": {
        "name": "cpu",
        "display_name": "CPU",
        "memory": 34359738368,
        "flops": {
            "bfloat16": null,
            "float32": null
        }
        },
        "hosts": [
        {
            "name": "balloon.jemoka.com",
            "cluster": {
            "name": "local",
            "root": "/Users/houjun/theseus",
            "work": "-",
            "log": null,
            "data": null,
            "checkpoints": null,
            "results": null,
            "status": null,
            "objects": null,
            "mount": null,
            "cache_size": null,
            "cache_dir": null,
            "all_squash": null
            },
            "resources": {
            "cpu": 1
            },
            "uv_groups": [],
            "env": {}
        }
        ],
        "total_chips": 1
    },
    "distributed": false
    }

    2026-09-08 21:02:55.534 | INFO     | theseus.training.base:__init__:191 - CONFIG | 
    {'batch_size': 512,
    'per_device_batch_size': 4,
    'total_tokens': 1000000000,
    'lr': 0.0003,
    'warmup_pct': 0.01,
    'decay_pct': 0.1,
    'validate': True,
    'evaluate': True,
    'analyze': True,
    'block_size': 512,
    'report_interval': 1,
    'checkpoint_interval': 1,
    'validation_interval': 512,
    'validation_steps': 2048}

    2026-09-08 18:31:44.956 | DEBUG    | theseus.training.base:__init__:192 - TRAINER | Initializing Topology
    2026-09-08 18:31:45.245 | DEBUG    | theseus.training.base:__init__:199 - TRAINER | Initializing Batch Config
    2026-09-08 18:31:45.246 | INFO     | theseus.training.base:_init_batch_config:241 - BATCHING | 4 batchsize/node * (1 local * 1 prox = 1 dp) * 128 accumulation = 512 batchsize
    2026-09-08 18:31:45.246 | INFO     | theseus.training.base:_init_batch_config:250 - STEPS | 488192 micro batches // 128 accumulation = 3814 steps
    2026-09-08 18:31:45.247 | INFO     | theseus.training.base:_init_batch_config:256 - TOKENS | 3814 steps * 512 batchsize * 512 blocksize = 999817216 tokens
    2026-09-08 18:31:45.247 | DEBUG    | theseus.training.base:__init__:201 - TRAINER | Hydrating Data Loaders
    2026-09-08 18:31:45.266 | DEBUG    | theseus.job:setup:218 - JOB train-gpt | building state
    2026-09-08 18:31:45.266 | DEBUG    | theseus.job:setup:240 - JOB train-gpt | starting anew
    2026-09-08 18:31:45.267 | DEBUG    | theseus.training.base:initialize:463 - TRAINER | Initializing Model Parameters, State, and Optimizers
    2026-09-08 18:31:49.667 | INFO     | theseus.training.base:_init_counters_and_eval:301 - MODEL | Total Parameters: 13.63m
    2026-09-08 18:31:49.668 | DEBUG    | theseus.inference.base:from_trainer:149 - INFERENCE | from_trainer replicas=1 local_replicas=1 per_device_batch_size=4 block_size=512
    2026-09-08 18:31:49.795 | INFO     | theseus.training.base:_init_counters_and_eval:316 - GPT(
        # attributes
        param_dtype = 'float32'
        activation_dtype = 'bfloat16'
        n_layers = 4
        n_embd = 128
        rope = True
        block_size = 512
        dropout = 0.0
        vocab_size = 100288
        instrument_residual = False
    )
    2026-09-08 18:31:49.796 | INFO     | theseus.job:__call__:249 - JOB train-gpt | starting
    2026-09-08 18:31:49.796 | DEBUG    | theseus.job:__call__:250 - JOB train-gpt | pre-start sync
    2026-09-08 18:31:49.797 | INFO     | theseus.job:__call__:260 - JOB train-gpt | synchronized and starting with nodeid=Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MA
    2026-09-08 18:31:49.910 | INFO     | theseus.training.profiler:start:142 - PROFILE | http://wroclaw:9000 | process=0 | JAX RPC port=9001
    2026-09-08 18:31:49.911 | INFO     | theseus.training.base:train:917 - BEGIN TRAINING
    2026-09-08 18:31:49.956 | DEBUG    | theseus.training.base:train:929 - DATA | 1 | START
    2026-09-08 18:31:50.000 | DEBUG    | theseus.training.base:train:931 - DATA | 1 | PLACED
    2026-09-08 18:32:06.117 | INFO     | theseus.training.base:train:951 - TRAIN | XLA cost analysis (loops counted once): {'bytes accessed46{}': 4.0, 'utilization123{}': 2.0, 'utilization153{}': 2.0, 'utilization24{}': 6.0, 'utilization157{}': 2.0, 'bytes accessedout{10}': 512.0, 'bytes accessed36{}': 4.0, 'utilization105{}': 4.0, 'utilization67{}': 4.0, 'utilization0{}': 1457.0, 'bytes accessedout{4}': 8389120.0, 'bytes accessed0{}': 4029964032.0, 'utilization90{}': 4.0, 'bytes accessedout{1}': 555433472.0, 'utilization108{}': 2.0, 'bytes accessed26{}': 4.0, 'utilization141{}': 2.0, 'utilization10{}': 10.0, 'bytes accessed24{}': 4.0, 'utilization39{}': 5.0, 'utilization26{}': 5.0, 'utilization175{}': 1.0, 'utilization178{}': 1.0, 'bytes accessed34{}': 4.0, 'utilization53{}': 4.0, 'bytes accessed35{}': 4.0, 'utilization78{}': 4.0, 'utilization134{}': 2.0, 'utilization92{}': 4.0, 'bytes accessed45{}': 4.0, 'utilization12{}': 10.0, 'utilization116{}': 2.0, 'utilization7{}': 66.0, 'bytes accessed15{}': 4194308.0, 'utilization55{}': 4.0, 'bytes accessed16{}': 32772.0, 'utilization152{}': 2.0, 'utilization159{}': 1.0, 'utilization94{}': 4.0, 'utilization100{}': 4.0, 'bytes accessedout{25}': 512.0, 'utilization115{}': 2.0, 'utilization118{}': 2.0, 'utilization34{}': 5.0, 'utilization20{}': 6.0, 'bytes accessed18{}': 3145732.0, 'utilization77{}': 4.0, 'bytes accessed28{}': 4.0, 'utilization63{}': 4.0, 'utilization170{}': 1.0, 'bytes accessed38{}': 4.0, 'utilization133{}': 2.0, 'bytes accessed25{}': 4.0, 'utilization49{}': 5.0, 'utilization36{}': 5.0, 'bytes accessed5{}': 67144704.0, 'utilization167{}': 1.0, 'utilization88{}': 4.0, 'utilization126{}': 2.0, 'utilization151{}': 2.0, 'utilization1{}': 593.0, 'utilization22{}': 6.0, 'flops': 126865645568.0, 'utilization110{}': 2.0, 'utilization65{}': 4.0, 'utilization144{}': 2.0, 'utilization51{}': 4.0, 'bytes accessed44{}': 4.0, 'bytes accessedout{13}': 512.0, 'utilization31{}': 5.0, 'utilization173{}': 1.0, 'utilization47{}': 5.0, 'utilization121{}': 2.0, 'bytes accessed11{}': 1028.0, 'utilization70{}': 4.0, 'utilization139{}': 2.0, 'bytes accessed20{}': 4.0, 'bytes accessed50{}': 4.0, 'utilization86{}': 4.0, 'utilization99{}': 4.0, 'utilization166{}': 1.0, 'bytes accessed29{}': 4.0, 'utilization19{}': 6.0, 'utilization114{}': 2.0, 'bytes accessedout{9}': 512.0, 'bytes accessed39{}': 4.0, 'utilization33{}': 5.0, 'utilization58{}': 4.0, 'utilization72{}': 4.0, 'utilization150{}': 2.0, 'bytes accessed40{}': 4.0, 'bytes accessed30{}': 4.0, 'utilization3{}': 109.0, 'utilization132{}': 2.0, 'bytes accessed19{}': 4.0, 'utilization147{}': 2.0, 'bytes accessed48{}': 4.0, 'utilization35{}': 5.0, 'bytes accessed10{}': 548868.0, 'bytes accessedout{20}': 512.0, 'utilization74{}': 4.0, 'bytes accessed49{}': 4.0, 'bytes accessedout{14}': 512.0, 'bytes accessed3{}': 144790816.0, 'utilization131{}': 2.0, 'bytes accessed4{}': 65047576.0, 'bytes accessedout{19}': 512.0, 'utilization14{}': 10.0, 'bytes accessed47{}': 4.0, 'utilization165{}': 1.0, 'utilization168{}': 1.0, 'utilization41{}': 5.0, 'utilization113{}': 2.0, 'utilization57{}': 4.0, 'utilization80{}': 4.0, 'utilization96{}': 4.0, 'utilization106{}': 4.0, 'utilization29{}': 5.0, 'bytes accessed27{}': 4.0, 'utilization6{}': 66.0, 'utilization16{}': 10.0, 'utilization43{}': 5.0, 'bytes accessedout{22}': 512.0, 'bytes accessedout{2}': 65872896.0, 'utilization68{}': 4.0, 'utilization142{}': 2.0, 'bytes accessed37{}': 4.0, 'utilization82{}': 4.0, 'bytes accessedout{16}': 512.0, 'utilization149{}': 2.0, 'bytes accessed8{}': 59796484.0, 'utilization176{}': 1.0, 'utilization124{}': 2.0, 'bytes accessed17{}': 8.0, 'utilization45{}': 5.0, 'bytes accessed': 10742396928.0, 'utilization160{}': 1.0, 'utilization84{}': 4.0, 'bytes accessedout{}': 4003990784.0, 'utilization164{}': 1.0, 'utilization11{}': 10.0, 'utilization112{}': 2.0, 'utilization119{}': 2.0, 'utilization27{}': 5.0, 'utilization50{}': 5.0, 'utilization146{}': 2.0, 'utilization66{}': 4.0, 'utilization93{}': 4.0, 'bytes accessed7{}': 4202555.0, 'utilization182{}': 1.0, 'bytes accessed42{}': 4.0, 'bytes accessed9{}': 6299653.0, 'utilization13{}': 10.0, 'utilization130{}': 2.0, 'utilization79{}': 4.0, 'utilization148{}': 2.0, 'utilization52{}': 4.0, 'bytes accessed23{}': 4.0, 'bytes accessed2{}': 2246107136.0, 'utilization95{}': 4.0, 'utilization127{}': 2.0, 'utilization38{}': 5.0, 'utilization15{}': 10.0, 'bytes accessedout{23}': 512.0, 'utilization163{}': 1.0, 'utilization111{}': 2.0, 'utilization54{}': 4.0, 'bytes accessedout{17}': 512.0, 'utilization81{}': 4.0, 'bytes accessed13{}': 20.0, 'utilization145{}': 2.0, 'bytes accessed33{}': 4.0, 'utilization156{}': 2.0, 'bytes accessedout{0}': 213844928.0, 'bytes accessed41{}': 4.0, 'utilization21{}': 6.0, 'utilization104{}': 4.0, 'utilization181{}': 1.0, 'utilization37{}': 5.0, 'bytes accessed31{}': 4.0, 'utilization60{}': 4.0, 'utilization2{}': 172.0, 'bytes accessed21{}': 4.0, 'utilization89{}': 4.0, 'utilization76{}': 4.0, 'utilization140{}': 2.0, 'utilization9{}': 14.0, 'utilization174{}': 1.0, 'utilization122{}': 2.0, 'utilization23{}': 6.0, 'bytes accessedout{11}': 512.0, 'utilization129{}': 2.0, 'utilization48{}': 5.0, 'bytes accessed32{}': 4.0, 'bytes accessed22{}': 4.0, 'utilization62{}': 4.0, 'bytes accessed12{}': 4194308.0, 'utilization25{}': 6.0, 'utilization155{}': 2.0, 'utilization158{}': 1.0, 'utilization103{}': 4.0, 'utilization64{}': 4.0, 'utilization91{}': 4.0, 'utilization137{}': 2.0, 'transcendentals': 290394880.0, 'bytes accessedout{7}': 512.0, 'utilization180{}': 1.0, 'utilization30{}': 5.0, 'bytes accessedout{21}': 512.0, 'utilization162{}': 1.0, 'utilization87{}': 4.0, 'bytes accessedout{15}': 512.0, 'utilization73{}': 4.0, 'utilization177{}': 1.0, 'utilization169{}': 1.0, 'utilization125{}': 2.0, 'utilization128{}': 2.0, 'utilization4{}': 79.0, 'utilization59{}': 4.0, 'utilization46{}': 5.0, 'utilization107{}': 4.0, 'utilization32{}': 5.0, 'utilization161{}': 1.0, 'utilization98{}': 4.0, 'utilization143{}': 2.0, 'utilization18{}': 9.0, 'bytes accessedout{6}': 512.0, 'bytes accessedout{3}': 9179136.0, 'utilization102{}': 4.0, 'bytes accessedout{12}': 512.0, 'utilization109{}': 2.0, 'utilization75{}': 4.0, 'utilization136{}': 2.0, 'utilization61{}': 4.0, 'bytes accessed6{}': 1081556.0, 'utilization97{}': 4.0, 'utilization172{}': 1.0, 'utilization179{}': 1.0, 'utilization120{}': 2.0, 'utilization17{}': 10.0, 'utilization40{}': 5.0, 'bytes accessedout{8}': 512.0, 'utilization154{}': 2.0, 'bytes accessed43{}': 4.0, 'utilization83{}': 4.0, 'utilization117{}': 2.0, 'utilization69{}': 4.0, 'bytes accessedout{5}': 8389120.0, 'utilization56{}': 4.0, 'utilization42{}': 5.0, 'utilization101{}': 4.0, 'utilization85{}': 4.0, 'utilization135{}': 2.0, 'utilization138{}': 2.0, 'utilization28{}': 5.0, 'bytes accessedout{24}': 512.0, 'bytes accessed1{}': 1021561472.0, 'bytes accessedout{18}': 512.0, 'bytes accessed14{}': 32772.0, 'utilization171{}': 1.0, 'utilization5{}': 74.0, 'utilization44{}': 5.0, 'utilization71{}': 4.0, 'optimal_seconds': -1.0, 'utilization8{}': 66.0}
        2026-09-08 18:32:06.127 | DEBUG    | theseus.training.base:train:970 - COMPUTATION | 1 | FINISHED
        2026-09-08 18:32:08.217 | INFO     | theseus.training.base:train:998 - TRAIN | 1/3814 | loss 11.530526161193848
        2026-09-08 18:32:08.218 | DEBUG    | theseus.training.base:train:1007 - STEP | 1 | {'train/lr': 1.8631602870300412e-05, 'train/mfu': 0.006672818739071294, 'train/tokens': 262144, 'train/loss': 11.530526161193848, 'train/grad_norm': 2.4589524269104004}
    2026-09-08 18:32:08.278 | DEBUG    | theseus.job:save:353 - CHECKPOINT | starting save
    2026-09-08 18:32:08.284 | DEBUG    | theseus.checkpoint:save:40 - CKMGR | saving checkpoint to /home/houjun/dispatch/root/objects/blobs/Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MA/checkpoint
    2026-09-08 18:32:08.285 | DEBUG    | theseus.checkpoint:save:42 - CKMGR | checkpointer ready
    2026-09-08 18:32:08.285 | DEBUG    | theseus.checkpoint:save:44 - CKMGR | attempting save
    2026-09-08 18:32:08.539 | DEBUG    | theseus.checkpoint:save:46 - CKMGR | save done on 0
    2026-09-08 18:32:08.541 | DEBUG    | theseus.job:save:355 - CHECKPOINT | saved training state
    2026-09-08 18:32:08.983 | DEBUG    | theseus.job:save:358 - CHECKPOINT | saved randomness
    2026-09-08 18:32:09.059 | DEBUG    | theseus.job:save:363 - CHECKPOINT | saved configuration
    2026-09-08 18:32:09.623 | DEBUG    | theseus.job:save:370 - CHECKPOINT | saved job spec
    2026-09-08 18:32:10.150 | INFO     | theseus.training.base:checkpoint:1058 - CHECKPOINT | saved checkpoint at Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MA at step 1
    2026-09-08 18:32:10.151 | DEBUG    | theseus.training.base:train:929 - DATA | 2 | START
    2026-09-08 18:32:10.153 | DEBUG    | theseus.training.base:train:931 - DATA | 2 | PLACED
    2026-09-08 18:32:10.160 | DEBUG    | theseus.training.base:train:970 - COMPUTATION | 2 | FINISHED
    2026-09-08 18:32:12.015 | INFO     | theseus.training.base:train:998 - TRAIN | 2/3814 | loss 11.53039264678955
    2026-09-08 18:32:12.016 | DEBUG    | theseus.training.base:train:1007 - STEP | 2 | {'train/lr': 3.4263182897120714e-05, 'train/mfu': 0.03236841618715644, 'train/tokens': 524288, 'train/loss': 11.53039264678955, 'train/grad_norm': 2.4105522632598877}
    2026-09-08 18:32:12.042 | DEBUG    | theseus.job:save:353 - CHECKPOINT | starting save
    2026-09-08 18:32:12.050 | DEBUG    | theseus.checkpoint:save:40 - CKMGR | saving checkpoint to /home/houjun/dispatch/root/objects/blobs/Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ/checkpoint
    2026-09-08 18:32:20.544 | DEBUG    | theseus.checkpoint:save:42 - CKMGR | checkpointer ready
    2026-09-08 18:32:20.545 | DEBUG    | theseus.checkpoint:save:44 - CKMGR | attempting save
    2026-09-08 18:32:20.595 | DEBUG    | theseus.checkpoint:save:46 - CKMGR | save done on 0
    2026-09-08 18:32:20.596 | DEBUG    | theseus.job:save:355 - CHECKPOINT | saved training state
    2026-09-08 18:32:21.362 | DEBUG    | theseus.job:save:358 - CHECKPOINT | saved randomness
    2026-09-08 18:32:21.507 | DEBUG    | theseus.job:save:363 - CHECKPOINT | saved configuration
    2026-09-08 18:32:53.282 | DEBUG    | theseus.job:save:370 - CHECKPOINT | saved job spec
    2026-09-08 18:32:53.676 | INFO     | theseus.training.base:checkpoint:1058 - CHECKPOINT | saved checkpoint at Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ at step 2
    2026-09-08 18:32:53.677 | DEBUG    | theseus.training.base:train:929 - DATA | 3 | START
    2026-09-08 18:32:53.680 | DEBUG    | theseus.training.base:train:931 - DATA | 3 | PLACED
    2026-09-08 18:32:53.686 | DEBUG    | theseus.training.base:train:970 - COMPUTATION | 3 | FINISHED
    2026-09-08 18:32:55.542 | INFO     | theseus.training.base:train:998 - TRAIN | 3/3814 | loss 11.518386840820312
    2026-09-08 18:32:55.543 | DEBUG    | theseus.training.base:train:1007 - STEP | 3 | {'train/lr': 4.9894762923941016e-05, 'train/mfu': 0.002823700688010997, 'train/tokens': 786432, 'train/loss': 11.518386840820312, 'train/grad_norm': 2.4760217666625977}
    2026-09-08 18:32:55.576 | DEBUG    | theseus.job:save:353 - CHECKPOINT | starting save
    2026-09-08 18:32:55.583 | DEBUG    | theseus.checkpoint:save:40 - CKMGR | saving checkpoint to /home/houjun/dispatch/root/objects/blobs/Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6Mg/checkpoint
    2026-09-08 18:32:55.584 | DEBUG    | theseus.checkpoint:save:42 - CKMGR | checkpointer ready
    2026-09-08 18:32:55.585 | DEBUG    | theseus.checkpoint:save:44 - CKMGR | attempting save
    2026-09-08 18:32:55.625 | DEBUG    | theseus.checkpoint:save:46 - CKMGR | save done on 0
    2026-09-08 18:32:55.626 | DEBUG    | theseus.job:save:355 - CHECKPOINT | saved training state
    2026-09-08 18:32:55.820 | DEBUG    | theseus.job:save:358 - CHECKPOINT | saved randomness
    2026-09-08 18:32:56.410 | DEBUG    | theseus.job:save:363 - CHECKPOINT | saved configuration
    2026-09-08 18:32:57.321 | DEBUG    | theseus.job:save:370 - CHECKPOINT | saved job spec

    ...
    ```
    
    also you really shouldn't be doing this on a laptop.

Actually, there's one problem. How'd I parse these logs? For post-hoc analysis, the best place to do this is by [navigating to the next chapter](inspecting-runs.md). For during run analysis, we support logging with [comet](https://www.comet.com/) which after logging in you turn on via the `q.config.logging.remote=True` field.

Ready to move on? Let's do some [analysis](inspecting-runs.md)!
