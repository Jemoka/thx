# Remote training
Ok so let's say you've got a neat job configured.

```python
from omegaconf import OmegaConf
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "dont matter what's here now")
    q.config.architecture.n_embd = 128
```

Let's stick up to a box in the sky! First, let's introduce a new friend. They're called a `Combobulation`, which encodes a *recipe* plus a *resource request*. Whereas `quick` describes only the recipe that you want to make, `Combobulation` also constraints where you need to make the actual run happen.

To turn a `QuickJob` into a `Combobulation`, we ask for `.spec()`:


```python
from omegaconf import OmegaConf
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "dont matter what's here now")
    q.config.architecture.n_embd = 128
    
    print(q.spec())
```
???+ example "command output"
    ```
    steps=(Step(job=<class 'theseus.experiments.models.gpt.PretrainGPT'>, base=None, resume=False),) minimum_memory=None cpus=None gpus=None chips=() preferred_clusters=() forbidden_clusters=() sharding=ShardingPolicy(tp=1, fsdp=False, zero=True)
    ```

Very combobulatory indeed. You will notice that it now has a few fields that we previously don't have access to, largely describing hardware. Much like quick, we can configure these fields by applying operators to them.

The most common fields are 1) what GPUs to use 2) how many of them 3) how to use them (e.g., sharding). Let's put some values up for each one:

```python
with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "train-gpt")
    q.config.architecture.n_embd = 128
    
    c = (
        q
        .spec()
        .chip("h100") # use h100s
        .gpu(4) # 4 of em
        .shard(tp=2, fsdp=True, zero=True) # use this shardig plan
    )
    
    print(c)
```
???+ example "command output"
    ```
    steps=(Step(job=<class 'theseus.experiments.models.gpt.PretrainGPT'>, base=None, resume=False),) minimum_memory=None cpus=None gpus=4 chips=('h100',) preferred_clusters=() forbidden_clusters=() sharding=ShardingPolicy(tp=2, fsdp=True, zero=True)
    ```

The GPU and Chip fields are self explanatory. Acceptable values of the latter is given in [this page](../Design/chips.md). Let's spend a second to talk about `.shard()`. Theseus supports implementations that are morally similar to DeepSpeed Zero-1, FSDP, and Tensor Parallel, the exact details of which are available in [this page](../Design/sharding.md). You can configure these knobs through `.shard`.

We are almost ready to send things off to the machine gods. To do this, we will need three steps.

## Step 1: Formalize Our Run!
Easiest step first. Dump our Combobulation to a file:

```python
from omegaconf import OmegaConf

with quick("/Users/houjun/theseus") as q:
    q.build("gpt/train/pretrain", "train-gpt")
    q.config.architecture.n_embd = 128
    
    c = (
        q
        .spec()
        .chip("h100") # use h100s
        .gpu(4) # 4 of em
        .shard(tp=2, fsdp=True, zero=True) # use this shardig plan
    )

    OmegaConf.save(c.serialize(), "out.yaml")
```

## Step 2: Formalize Our Hardware!
`~/.theseus.yaml` is theseus's cluster information file. You use this file to tell theseus what computers are available to you. We support dispatching to plain SSH hosts, SLURM clusters, and Volcano K8s clusters.

Here's a very minimal configuration you can customize for your needs.

```yaml
clusters:
  mycoolcluster:
    root: /home/houjun/root # <- logically, what went into quick([here])
    work: /home/houjun/work # <- where we should copy Theseus source to
    log: /home/houjun/log # <- where should execution log go

hosts:
  mycoolcomputor:
    ssh: mycoolcomputor.jemoka.com # theseus will run `ssh mycoolcomputor.jemoka.com`
    cluster: mycoolcluster # <- one of the names above
    type: plain
    chips:
      h100: 8
    uv_groups: [all, cuda13] # <- *should be* (sorry) all + cuda12 OR all + cuda13

  mywarmcomputor:
    ssh: mywarmcomputor.jemoka.com # theseus will run `ssh mycoolcomputor.jemoka.com`
    cluster: mycoolcluster # <- one of the names above
    type: plain
    chips:
      h100: 8
    uv_groups: [all, cuda13] # <- *should be* (sorry) all + cuda12 OR all + cuda13

```

How do you deal with passwords and ssh user names and such? We defer this problem to `~/.ssh/config`, so any ssh-related configuration should just go there to the point where `ssh [host]` juts works:tm:.

As I said before, we support SSH, SLURM, and Volcano K8s. To configure those all, set environment variable, that type of thing, visit [this page](../How-to/remote-configuration/providers.md).

## wEEEEEEEEEE
Ok now we are ready to put the petal to the metal. In your terminal, run:

```
uv run theseus submit train-gpt out.yaml -p cool-project
```

theseus will ssh into each host, determine through appropriate mechanisms whether its occupied, and if not, submit the job there. Once it tells you what host it chose, ssh into it to the logs folder and have a field day!

