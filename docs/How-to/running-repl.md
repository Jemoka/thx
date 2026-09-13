# Interactive inspection

Start Jupyter in the environment that can access your run store and devices:

```bash
uv run --with jupyterlab jupyter lab
```

Use `theseus.quick.quick` inside the notebook to select checkpoints, build a
job, and inspect its model with `debug`. See [Inspecting runs](inspecting-runs.md)
for the browser interface. See [Analysis and plotting](../Design/plot.md) for
the Python inspection and plotting APIs.

Theseus no longer provides a `repl` command or live code synchronization.
For a remote notebook, allocate the machine through your cluster's normal
interactive workflow and forward the Jupyter port yourself. The notebook
process must have the same dependencies and store access as the job.
