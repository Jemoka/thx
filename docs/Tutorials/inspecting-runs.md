# Inspecting Runs
Ok so you did a training run well and good and you are very happy which is great for you but uhhh what are you going to do with it? what are the losses like? can you restore a checkpoint? I will say, this is a really cool part of theseus, but also some of the most non-standard machine learning-y things about this project. so, uuhuhhhh, buckle up?

## where's my run!
Here's a little secret: `quick` is a cursor over a database[^1] (badly) backed by clickhouse[^2]. It contains a neat little Python DSL which we can use to find *logs, runs, and checkpoints* in the same exact API. Let me show you what I mean. Let's get our run back:

```python
with quick("/Users/houjun/theseus") as q:
    print(q.find().spec(run="train-gpt").latest().all())
```
??? example "command output"
    ```
    [Node(name='general.default.train-gpt', nonce='523428', seq=0, parent=None),
     Node(name='general.default.train-gpt', nonce='949e88', seq=0, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=1, parent=None),
     Node(name='general.default.train-gpt', nonce='949e88', seq=1, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=2, parent=None),
     Node(name='general.default.train-gpt', nonce='949e88', seq=2, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=3, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=4, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=5, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=6, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=7, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=8, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=9, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=10, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=11, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=12, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=13, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=14, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=15, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=16, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=17, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=18, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=19, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=20, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=21, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=22, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=23, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=24, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=25, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=26, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=27, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=28, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=29, parent=None),
     Node(name='general.default.train-gpt', nonce='523428', seq=30, parent=None)]
    ```
    
    yours is going to look different than mine, but illustratively

huh. what are these `Node` things? You can think of every `Node` as "a moment something happened." Nodes have three fields you should know about:

- **name**: the name
- **nonce**: even if you launch two runs with the same name, this will be different
- **seq**: roughly, "step" just think about it as "every time something different happens", seq is incremented

and they can contain arbitrary stuff inside them: information (like loss), artifacts (like checkpoints), and the like. Now look back to your training transcript. It gave you a big ol' string:

```
synchronized and starting with nodeid=Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MA
```

this is the *starting node* of your run. we can get more info about it from the database!

```python
with quick("/Users/houjun/theseus") as q:
    print(q.find().node("Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MA").all())
```
???+ example "command output"
    ```
    [Node(name='general.default.train-gpt', nonce='949e88', seq=0, parent=None)]
    ```

Cool! This gave us an ID: the `nonce` of our run is `949e88`. In theseus, this means that every checkpoint, loss, and so on derived from this run will be keyed by the same nonce. We can then do such amazing things as "get the loss over tokens" and then you can do amazing things like plot it.

```python
with quick("/Users/houjun/theseus") as q:
    loss_over_time = q.find().nonce("949e88").select(keys=["train/loss", "train/tokens"])
    print(loss_over_time)                   # ^ not all! select!

```
???+ example "command output"
    ```
    [{'train/loss': 11.530526161193848, 'train/tokens': 262144}, {'train/loss': 11.53039264678955, 'train/tokens': 524288}, {'train/loss': 11.518386840820312, 'train/tokens': 786432}]
    ```
> BTW: if you did `.select()` bare, expecting it to return every column to you, you will be sordidly disappointed because there will be too much data and everything will crash and burn since you have accidentally asked for the entire Jax initial profiling graph which is like a lot of things. So don't do that.

## time travel
    
At this point you may go "geez jack I didn't expect you asking me to build `wandb` out of seaborn also didn't you already say you had a [comet logging integration](./running.md#build-a-training-job)? why am I plotting myself?" 

First, I can't hear you.

Second, even if I did, I would have an answer. Which is that, well, yeah, you shouldn't be using this API to get basic lines. Remember my point about *checkpoints* being in the same API? So, we can do such things as **find the checkpoint with the lowest loss AND under 700k tokens AND grad norm larger than 1, and then resume training it.**

```python
with quick("/Users/houjun/theseus") as q:
    (
        q.find().nonce("949e88")
        .checkpoint() # <- the node must contain a checkpoint
        .where("train/tokens", "<=", 700_000)
        .where("train/grad_norm", ">=", 1)
        .sort("train/loss", ascending=False)
        # jee, jack, a bug  ^ in your docs?
        # nah, .resume() selects the LAST thing
        # the query returns to restore. so
        # descending restores the lowest loss
        .resume()
    )
    print(q.config)  # it just knows
    job = q.build() # yes, it also just knows
    job()
```
??? example "command output"
    ```
    {'architecture': {'dtype': {'param': 'float32', 'activation': 'bfloat16'}, 'n_embd': 128, 'n_layers': 4, 'bias': True, 'dropout': 0.0, 'n_head': 16, 'block_size': 512, 'rope': True, 'instrumentation': {'components': False, 'residual': False}, 'intermediate_size': -1, 'layer_norm_eps': 1e-05, 'vocab_size': 100288}, 'eval': {'length': -1}, 'tokenizer': {'backend': 'tiktoken', 'name': 'cl100k_base', 'huggingface': {'use_fast': True, 'use_remote_code': False}}, 'data': {'suffix': '', 'fineweb': {'snapshot': 'CC-MAIN-2022-21'}}, 'optimization': {'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.95, 'lr': 0.0003, 'warmup_pct': 0.005, 'decay_pct': 0.01, 'final_lr_frac': 0.01}, 'training': {'batch_size': 512, 'per_device_batch_size': 4, 'tokens': 1000000000, 'warmup_pct': 0.01, 'decay_pct': 0.1, 'validation': True, 'evaluate': True, 'analyze': True, 'validation_steps': 2048}, 'logging': {'report_interval': 1, 'checkpoint_interval': 1, 'validation_interval': 512, 'remote': False}}
    
    2026-09-08 19:21:44.775 | DEBUG    | theseus.job:from_node:618 - CHECKPOINT | restored job spec from checkpoint
    2026-09-08 19:21:46.714 | INFO     | theseus.training.base:__init__:187 - TOPOLOGY | 
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
        "name": "l40",
        "display_name": "Nvidia L40",
        "memory": 51539607552,
        "flops": {
            "bfloat16": 181.05,
            "float32": 90.5
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
        "name": "l40",
        "display_name": "Nvidia L40",
        "memory": 51539607552,
        "flops": {
            "bfloat16": 181.05,
            "float32": 90.5
        }
        },
        "hosts": [
        {
            "name": "wroclaw",
            "cluster": {
            "name": "local",
            "root": "/home/houjun/dispatch/root",
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
            "l40": 1
            },
            "uv_groups": [],
            "env": {}
        }
        ],
        "total_chips": 1
    },
    "distributed": false
    }

    2026-09-08 19:21:46.717 | INFO     | theseus.training.base:__init__:191 - CONFIG | 
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

    2026-09-08 19:21:46.717 | DEBUG    | theseus.training.base:__init__:192 - TRAINER | Initializing Topology
    2026-09-08 19:21:47.008 | DEBUG    | theseus.training.base:__init__:199 - TRAINER | Initializing Batch Config
    2026-09-08 19:21:47.009 | INFO     | theseus.training.base:_init_batch_config:241 - BATCHING | 4 batchsize/node * (1 local * 1 prox = 1 dp) * 128 accumulation = 512 batchsize
    2026-09-08 19:21:47.010 | INFO     | theseus.training.base:_init_batch_config:250 - STEPS | 488192 micro batches // 128 accumulation = 3814 steps
    2026-09-08 19:21:47.010 | INFO     | theseus.training.base:_init_batch_config:256 - TOKENS | 3814 steps * 512 batchsize * 512 blocksize = 999817216 tokens
    2026-09-08 19:21:47.010 | DEBUG    | theseus.training.base:__init__:201 - TRAINER | Hydrating Data Loaders
    2026-09-08 19:21:47.031 | DEBUG    | theseus.job:setup:218 - JOB train-gpt | building state
    2026-09-08 19:21:47.031 | DEBUG    | theseus.job:setup:222 - JOB train-gpt | restoring state baseof=Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ
    2026-09-08 19:21:50.667 | DEBUG    | theseus.checkpoint:restore:72 - CKMGR | restoring checkpoint from /home/houjun/dispatch/root/objects/blobs/Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ/checkpoint
    2026-09-08 19:21:50.669 | DEBUG    | theseus.checkpoint:restore:80 - CKMGR | partial restore enabled
    2026-09-08 19:21:50.796 | DEBUG    | theseus.checkpoint:restore:98 - CKMGR | checkpoint matches target; restoring without transforms
    2026-09-08 19:21:53.984 | INFO     | theseus.checkpoint:restore:144 - CKMGR | restored checkpoint from /home/houjun/dispatch/root/objects/blobs/Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ/checkpoint
    2026-09-08 19:21:53.986 | INFO     | theseus.training.base:_init_counters_and_eval:301 - MODEL | Total Parameters: 13.63m
    2026-09-08 19:21:53.987 | DEBUG    | theseus.inference.base:from_trainer:149 - INFERENCE | from_trainer replicas=1 local_replicas=1 per_device_batch_size=4 block_size=512
    2026-09-08 19:21:54.151 | INFO     | theseus.training.base:_init_counters_and_eval:316 - GPT(
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
    2026-09-08 19:21:54.152 | INFO     | theseus.job:__call__:249 - JOB train-gpt | starting
    2026-09-08 19:21:54.152 | DEBUG    | theseus.job:__call__:250 - JOB train-gpt | pre-start sync
    2026-09-08 19:21:54.153 | INFO     | theseus.job:__call__:260 - JOB train-gpt | synchronized and starting with nodeid=Z2VuZXJhbC5kZWZhdWx0LnRyYWluLWdwdDo5NDllODg6MQ
    2026-09-08 19:21:54.272 | INFO     | theseus.training.profiler:start:142 - PROFILE | http://wroclaw:9000 | process=0 | JAX RPC port=9001
    2026-09-08 19:21:54.273 | INFO     | theseus.training.base:train:917 - BEGIN TRAINING
    2026-09-08 19:21:54.318 | DEBUG    | theseus.training.base:train:929 - DATA | 3 | START
    2026-09-08 19:21:54.347 | DEBUG    | theseus.training.base:train:931 - DATA | 3 | PLACED
    ...

    ```

Nice. You just restored a training run. This will work even if you copied (or NFS'd) your root folder to a different machine, to a different topology, with a different number of GPUs (more OR less), or on a different planet[^3].

## time travel debugging
Ok so all the restore business is nice and all, but what if i wanted moooaarrrrr. I want to know what batch was used during this forward pass, and I want to know what layer 1's MLP inputs look like.

dwabit, I gochu. these same semantics also expose a series of nice debugging APIs that you can take advantage of. Let's walk through each of these desiderata.

### what batch was that step again?
You can just ask for it

```python
with quick("/Users/houjun/theseus") as q:
    (
        q.find().nonce("949e88")
        .checkpoint()
        .where("train/tokens", "<=", 700_000)
        .where("train/grad_norm", ">=", 1)
        .sort("train/loss", ascending=False)
        .resume()
    )

    job = q.build()
    trainer = job.create()
    print(trainer.batch())
```
??? example "command output"
    ```
    {'x': array([[ 3756,   323,  1005, ...,   311,  6227,   904],
        [  994,   279, 10873, ...,   304,   701, 12818],
        [93032, 58076,   304, ...,  1611,   650, 20000],
        ...,
        [ 3325,  1772,   389, ...,  1789, 32917, 27711],
        [  753,   311,  2586, ..., 17065,  9131,   627],
        [ 3666,   365, 12429, ...,   527,  9966,    11]], shape=(512, 512)), 'y': array([[  323,  1005,  1124, ...,  6227,   904,   892],
        [  279, 10873,  2543, ...,   701, 12818,    11],
        [58076,   304,   279, ...,   650, 20000,    11],
        ...,
        [ 1772,   389,  3674, ..., 32917, 27711,  1501],
        [  311,  2586,    11, ...,  9131,   627,  4054],
        [  365, 12429,  6959, ...,  9966,    11,   439]], shape=(512, 512)), 'padding_mask': array([[ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        ...,
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True],
        [ True,  True,  True, ...,  True,  True,  True]], shape=(512, 512))}
    ```
    
Lots to learn from this one example. First, `.build()` => `.create()` is a bit semantically confusing but do two different tasks.

- `.build()`: please give me a `QuickJob`, which is a fully-serialized job description from all the shenanigans from above
- `.create()`: please actually load the state from the database, and give me an object representing the underlying trainer

Second, `trainer.batch()` will keep returning you the same batch. This is becasue the dataloader (and everything else) is *keyed by node*. What this means is that every node will always represent one unique state during training. So, if you wanted the next batch the dataloader would've fed, you would do `trainer.tick()` (please get me the next node) => `trainer.batch()`.

### what in the residual?
Let's dial up the determinism knob to 11[^4]. What if I don't just want the batch that was fed in, but to replay the computation in the middle of a state? 

We will first have to figure out what to replay. Say, I care about the MLP inputs, we can ask not only for which MLP exist in a node, but what inputs they got and step through it. Here's what I mean:

```python
from theseus.model.layers.mlp import MLP 
with quick("/Users/houjun/theseus") as q:
    (
        q.find().nonce("949e88")
        .checkpoint()
        .where("train/tokens", "<=", 700_000)
        .where("train/grad_norm", ">=", 1)
        .sort("train/loss", ascending=False)
        .resume()
    )

    trainer = q.build().create()
    print(trainer.find(MLP))
```
???+ example "command output"
    ```python
    [('blocks_0', 'mlp'), ('blocks_1', 'mlp'), ('blocks_2', 'mlp'), ('blocks_3', 'mlp')]
    ```
    (remember how our model only has like, 4 layers? its so cute)

and say I really cared about the block 2 MLP input. For this part to make sense, you will have to take a look roughly at how the MLP is implemented:

??? note "theseus MLP implementation"
    ```python
    class MLP(Module):
        ...    

        @nn.compact
        def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
            x = self.c_fc(x)
            x = jax.nn.gelu(x)
            x = self.c_proj(x)

            if not deterministic:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

            return x
    ```

With this in your head, we can do some tricks. For instance, we can ask what the input to this layer is:

```python
from theseus.model.layers.mlp import MLP 
with quick("/Users/houjun/theseus") as q:
    (
        q.find().nonce("949e88")
        .checkpoint()
        .where("train/tokens", "<=", 700_000)
        .where("train/grad_norm", ">=", 1)
        .sort("train/loss", ascending=False)
        .resume()
    )

    trainer = q.build().create()
    layer, inputs = trainer.debug()
    print(inputs.x) # <- the actual argument name to the __call__ function
    layer.close() # <- free temporaries
```
??? example "command output"
    ```
    [[[0.714844 -0.503906 0.636719 ... -0.347656 -1.375 -0.503906]
    [-0.474609 -1.3125 1.03906 ... 0.380859 -0.224609 0.302734]
    [-0.257812 -1.88281 0.388672 ... -0.730469 0.828125 -0.498047]
    ...
    [1.13281 0.625 -0.378906 ... 0.241211 -0.0356445 -0.126953]
    [0.625 -1.23438 0.154297 ... -0.683594 -2 -0.617188]
    [-0.132812 -0.839844 0.160156 ... 0.40625 -1.16406 0.128906]]

    [[-0.757812 -0.119629 -1.60938 ... 0.855469 0.511719 0.8125]
    [-2.85938 -0.546875 -1.26562 ... 0.490234 -0.988281 -0.171875]
    [-1.39844 -0.742188 -0.648438 ... 0.679688 -0.225586 -0.0668945]
    ...
    [1.91406 -0.726562 -1 ... -0.181641 -0.478516 -1.67969]
    [0.777344 0.753906 -0.449219 ... -0.699219 -1.27344 0.835938]
    [-0.820312 -0.136719 -1.08594 ... -0.108887 -1.5 1.04688]]

    [[-0.429688 -1 0.792969 ... 0.337891 1.09375 -0.523438]
    [0.507812 -1.08594 -0.53125 ... 0.832031 -0.894531 0.10791]
    [1.49219 -0.219727 -1.07812 ... 0.441406 0.429688 -1.21875]
    ...
    [0.0471191 -1.38281 -0.222656 ... -0.210938 -0.296875 -0.765625]
    [-1.97656 0.898438 0.110352 ... 2.8125 -0.0410156 2.26562]
    [-1.53125 0.320312 1.70312 ... 0.714844 0.894531 0.179688]]

    [[1.6875 0.259766 1.19531 ... -1.09375 -0.120605 1.89844]
    [1.04688 -0.283203 1.375 ... -0.839844 -0.394531 1.53906]
    [0.859375 -0.425781 1.02344 ... -0.960938 -0.59375 0.178711]
    ...
    [0.898438 -1.96875 1.46094 ... 0.808594 -1.89844 -0.90625]
    [0.0578613 -1.08594 -0.667969 ... 0.339844 0.052002 0.414062]
    [-1.95312 -0.792969 -1.28125 ... 1.59375 -0.223633 -0.00343323]]]
    ```

or, what the upproj returns (or its shape).

```python
from theseus.model.layers.mlp import MLP 
with quick("/Users/houjun/theseus") as q:
    (
        q.find().nonce("949e88")
        .checkpoint()
        .where("train/tokens", "<=", 700_000)
        .where("train/grad_norm", ">=", 1)
        .sort("train/loss", ascending=False)
        .resume()
    )

    trainer = q.build().create()
    layer, inputs = trainer.debug(("blocks_2", "mlp"))
    print(layer.c_fc(inputs.x).shape)

    layer.close() # <- free temporaries
```
???+ example "command output"
    ```
    2026-09-08 19:48:13.514 | DEBUG    | theseus.model.debug:_intercept:226 - DEBUG | blocks_2/mlp/c_fc/bias | reused parameter shape=(512,)
    2026-09-08 19:48:13.514 | DEBUG    | theseus.model.debug:_intercept:226 - DEBUG | blocks_2/mlp/c_fc/kernel | reused parameter shape=(128, 512)

    (4, 512, 512)
    ```

Oh look, what's that, a log line! Here, theseus is telling you that, given what it knows of the checkpoint, the only thing that would match the shape of the thing you asked for is probably `blocks_2/mlp/c_fc/kernel` and so it restored it using Jax eager mode blackmagic and now you can call it as if you are writing PyTorch.

Neat. You can make graphs this way for presentations and it absolutely slaps.

## notebook users
BTW, if you are in a Jupyter notebook I know that context managers is a bad day. I have an alternative spelling of `quick` you may want to know about:

```python
from theseus.quick import quick
with quick("/path/to/place") as q:
    ...
```

is synonymous with

```python
from theseus.quick import init
q = init("/path/to/place")
...
q.close()
```

Alright, with that out of the way, let's recognize a key elephant in the room: we've been training neural networks on a laptop... Let's uhh fix that by exploring [remote disptach](./remote.md).

[^1]: [andy pavlo](https://www.cs.cmu.edu/~pavlo/) swoons
[^2]: [andy pavlo](https://www.cs.cmu.edu/~pavlo/) falls over while sitting
[^3]: actual outcome is not covered by unit tests; but actually everything else on that list is. thanks codex
[^4]: [weeeeee](https://www.youtube.com/watch?v=72y2EC5fkcE)
