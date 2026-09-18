# Sharp Edges
theseus has a lot of sharp edges which generally boils down to "what Jack thought was useful is implemented, everything else that doesn't matter is ignored." but I realize a lot of these is not clear.

## The `PMD` dataloader do not do document masking.
Yup, that's right. I'm sorry. You can implement it, but I don't. But we do append `<eos>` tokens though. If you implement this you can do a PR.

## MoEs use naive shuffles.
I'll fix this eventually, but right now if you look at how the residuals are permuted for MoEs the answer is grossly and I'm sorry. I'm afraid of Pallas but I'll grow up soon.

## Autobatch doesn't work on multi-host settings.
Set a `per_device_batch_size` for human-intelligence driven batching.

## Reporting training progress blocks training.
Sorry. Just log less. I didn't want to gross the trainer to make another bounding queue.

## Plotting blocks training.
See above. Though I'll try to fix this eventually because actually theseus v0 had a better solution to this.

## Pipeline or expert parallelism?
I wish I command enough GPUs for this to matter. But I don't. But if you want to give me some you can [email me about it](mailto:hi@jemoka.com) and I'll buy you dinner. On the occasion that I suddenly have a lot of compute which happens from time to time I'll put stuff in.

## Analysis and Evaluation Runs 1-4 is VERY SLOW
Sorry, its doing JIT compilation, then three runs of profiling-guided latency estimation (PGLE), and then a recompilation of the optimized graph. It will be a 2-order-magnitude slowdown. But it should be fine:tm:. Sorry. 

If you are really worried `THESEUS_DISABLE_OPTIMIZATIONS=1` in `env:` section of your dispatch or locally can disable PGLE and expensive optimizations but then you are also sad because the later runs are a bit slow. ORRRR you can just disable PGLE only and see what that buys you, which is `JAX_ENABLE_PGLE=false`.

## JAX can hang with GPUs showing 100% utilization.

If a multi-GPU run stops making progress while GPU utilization remains at 100%, try setting `NCCL_P2P_DISABLE=1` in the training process's environment before starting the run:

```bash
export NCCL_P2P_DISABLE=1
```

This disables NCCL's direct GPU-to-GPU transport over NVLink or PCIe and may reduce communication performance. It's a workaround to try, we haven't tried it for every JAX hang!

## [yours here]
Open an issue! There's more but I need to think of them.
