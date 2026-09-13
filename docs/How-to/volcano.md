# Volcano dispatch

Theseus submits the same execution document to Volcano as it does to SSH or
SLURM. You need a configured `kubectl`, an existing Volcano queue, a shared PVC,
and an image suitable for the generated bootstrap.

Configure `chips`, `gpus_per_node`, and `num_nodes` to describe
the capacity available to this provider. The solver checks that the queue is
open; Volcano determines when the requested resources can be scheduled.

```bash
uv run theseus submit experiment train.yaml --cluster batch --chip h100 -n 8 --cpu 16 --mem 64Gi
```

A temporary helper uploads the bootstrap and dispatch document to the PVC.
Worker pods then run the execution with matching CPU, memory, and GPU requests.
Multiple workers receive their JAX process indices and coordinator address
through Volcano's `env` and `svc` plugins. The shared PVC must be mountable by
all workers, and workers must be able to reach each other.

Submission prints the Volcano job name. Inspect it in the configured namespace:

```bash
kubectl get jobs.batch.volcano.sh -n training
kubectl get pods -n training
kubectl logs -n training POD_NAME
kubectl describe jobs.batch.volcano.sh -n training JOB_NAME
```

To cancel an execution, delete its Volcano job:

```bash
kubectl delete jobs.batch.volcano.sh -n training JOB_NAME
```

Publication and submission are idempotent for an identical dispatch. Reusing
an identity for different dispatch contents is rejected.
