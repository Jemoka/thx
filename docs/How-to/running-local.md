# Running Locally

## Step 1 — Generate a config

```bash
uv run theseus configure gpt/train/pretrain run.yaml
```

That writes a fully-populated `run.yaml` with all default values filled in. Open it up and you'll see every knob the job exposes.

You can also override values inline:

```bash
uv run theseus configure gpt/train/pretrain run.yaml training.per_device_batch_size=8 \
    architecture.n_layers=12
```

Planning to run on a specific chip? Bake the hardware request into the config now so you don't have to repeat it later:

```bash
uv run theseus configure gpt/train/pretrain run.yaml --chip h100 -n 4
```

## Step 2 — Run it

```bash
uv run theseus run my-gpt-run run.yaml ./output
```

- `my-gpt-run` is a human-readable name for this run (used for logging/checkpointing).
- `./output` is where checkpoints, logs, and results land.

Override config values at run time the same way:

```bash
uv run theseus run my-gpt-run run.yaml ./output \
    training.per_device_batch_size=4
```

If you want to attach the run to a project or group:

```bash
uv run theseus run my-gpt-run run.yaml ./output \
    --project my-project --group ablation-lr
```
