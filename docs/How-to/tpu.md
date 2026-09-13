# Running on TPU

The current dispatch providers are SSH, SLURM, and Volcano. The retired TPU
provider's VM creation and `--tpu-*` flags are no longer available.

On an already provisioned single-host TPU environment, install the TPU
dependency group and use the local runner:

```bash
uv sync --group tpu
uv run theseus run experiment train.yaml /path/to/results
```

`train.yaml` is the same execution document used by `submit`; generate it with
`theseus configure` as described in [Running locally](running-local.md).
The local runner uses the devices visible to JAX. It does not provision hosts
or coordinate a TPU pod launch. Multi-host deployment must arrange process
startup and distributed initialization before running the execution.
