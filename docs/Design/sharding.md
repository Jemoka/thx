# Sharding Design

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

## One model, several ways to divide the work

Training has several large objects: examples, model parameters, intermediate
activations, gradients, and optimizer state. They do not all need the same
layout. Different devices can process different examples, cooperate on one
matrix multiplication, or keep different pieces of the optimizer state.

theseus combines three choices: **data parallelism** divides examples,
**tensor parallelism (TP)** divides model computation, and **ZeRO/FSDP** divide
state that would otherwise be duplicated across data-parallel replicas. The
model still sees global arrays with their original shapes. Sharding describes
which device owns which part of an array and what communication is needed to
operate on it.

## What an axis means

A **tensor axis** is a dimension of an array: vocabulary entries, input features,
output features, or examples in a batch. A **mesh axis** is a dimension of a grid
of devices. A sharding rule connects the two: divide this tensor dimension among
the devices along that mesh dimension.

theseus uses two physical mesh axes:

| Mesh axis | Meaning | What devices along it do |
| --- | --- | --- |
| `batch` | Data-parallel replicas | Process different examples. They can also share responsibility for storing parameters and optimizer state. |
| `shard` | Tensor-parallel devices within a replica | Cooperate on the same examples, each handling part of a tensor operation. |

For eight devices and `tp=2`, the mesh has shape `(batch=4, shard=2)`:

| Data replica | `shard=0` | `shard=1` | Examples |
| --- | --- | --- | --- |
| `batch=0` | Device 0 | Device 1 | Microbatch A |
| `batch=1` | Device 2 | Device 3 | Microbatch B |
| `batch=2` | Device 4 | Device 5 | Microbatch C |
| `batch=3` | Device 6 | Device 7 | Microbatch D |

Each row is one model replica in the computational sense: its two devices
jointly evaluate the model. The four rows process different examples. Calling
these rows replicas does not require every row to hold a complete persistent
copy of the parameters; FSDP changes their storage.

The mesh is arranged by process and device ID. `tp` must divide the number of
local devices so a tensor-parallel group stays within one host. The `batch`
axis can span hosts. Increasing `tp` on a fixed allocation reduces the number
of data replicas: eight devices with `tp=4` give two replicas, not eight.

This also affects batch accounting. theseus multiplies the effective
`per_device_batch_size` by the number of **data replicas**, then by gradient
accumulation steps to obtain the global batch size. Tensor-parallel devices
working on the same examples do not multiply the number of examples. In the
four-replica example, two examples per replica and eight accumulation steps
produce a global batch of 64 examples.

## Logical axes describe the model

A layer should know that a weight matrix has input and output feature dimensions
without needing to know how many GPUs will run it. theseus gives tensor
dimensions logical names, such as `n_embd`, `n_attn`, and `vocab`, through Flax
partition annotations. The model's `ShardingPlan` maps those names onto the
physical mesh.

For example, GPT's attention input projection is annotated with
`(n_embd, n_attn)`. Its TP rules leave `n_embd` unpartitioned and map `n_attn` to
`shard`. With `tp=2`, each device in a replica owns half the projection's output
columns. Both devices participate in computing the projection; later operations
may need communication to combine or redistribute their results.

An unpartitioned dimension retains its full extent on each device holding the
other dimensions' slices. An array is replicated across a mesh axis when none
of its dimensions is partitioned over that axis. A tensor with no partition
annotations remains replicated under these rules. Consequently, requesting TP
or FSDP does not automatically split every array in an arbitrary model.

There are two rule sets in a model's plan:

- **TP rules** describe the layouts used to execute the model.
- **ZeRO rules** describe additional partitions across `batch` for optimizer
  updates and, optionally, parameter storage.

When a model omits explicit ZeRO rules, theseus takes the logical dimensions
chosen for TP and maps them onto `batch` as well. Models can choose differently:
GPT, for example, keeps the vocabulary axis unpartitioned for TP but allows it
to be partitioned for ZeRO storage. A dimension can be split over both mesh axes;
with four batch replicas and two TP devices, that dimension has eight pieces.

The execution's `ShardingPolicy` chooses which of these layouts to use. This
separation lets the model describe sensible tensor boundaries while the same
model runs under different device counts and memory budgets.

## What TP, ZeRO, and FSDP each change

**TP divides the computation within each replica.** Devices hold different
parameter slices and compute parts of the model together. It can reduce the
parameter and activation memory needed by an individual device, but introduces
communication within model operations. The exact benefit depends on the model's
annotations and the operations being partitioned.

**ZeRO divides optimizer ownership across replicas.** Optimizers can maintain
large arrays alongside the parameters, such as Adam's moment estimates. When
`zero=True`, these arrays and the parameter updates are partitioned over
`batch`, in addition to any TP partitions. Each owner updates its assigned
piece using the gradient contributions required for that piece. Parameters can
still be replicated across `batch` for model execution and persistent storage.
This is the distinction between sharing optimizer state and sharing the stored
model itself.

**FSDP also divides persistent parameter storage across replicas.** With
`fsdp=True`, a replica does not keep all of its TP parameter slices locally
between computations. Parameters are gathered into the layouts needed to run
the model, then returned to their distributed storage layout. This saves
persistent parameter memory and adds communication to make parameters available
for computation. It does not remove the need for activations or temporary
working memory.

The policy defaults to `tp=1`, `zero=True`, and `fsdp=False`. FSDP requires
ZeRO in this implementation. With a mesh axis of size one, partitioning over
that axis provides no memory reduction.

## Why parameters have three layouts

The most memory-efficient way to keep a parameter between steps need not be
the layout its matrix multiplication or optimizer needs. theseus therefore
tracks three contracts explicitly:

| Phase | Parameter layout | Purpose |
| --- | --- | --- |
| Storage | TP; TP plus ZeRO partitions when FSDP is enabled | Keep the persistent training state in its chosen memory layout. |
| Forward/backward | TP | Give model operations their declared computational layouts. |
| Optimizer update | TP plus ZeRO partitions when ZeRO is enabled; otherwise TP | Put each parameter slice beside the optimizer state responsible for updating it. |

Consider an annotated weight of shape `(1024, 4096)` whose output dimension
uses `shard` for TP and `batch` for ZeRO, on the eight-device mesh above:

| Policy | Stored weight slice per device | Weight slice used for TP computation | Optimizer state for that weight, per device |
| --- | --- | --- | --- |
| TP only | `(1024, 2048)` | `(1024, 2048)` | Corresponds to `(1024, 2048)` |
| TP + ZeRO | `(1024, 2048)` | `(1024, 2048)` | Corresponds to `(1024, 512)` |
| TP + ZeRO + FSDP | `(1024, 512)` | `(1024, 2048)` | Corresponds to `(1024, 512)` |

These are placements of the same global weight, not changes to its model shape.
The optimizer column describes parameter-shaped state; scalar counters or
optimizer-specific arrays can have different shapes and layouts.

## What happens during a training step

The step begins with parameters and optimizer state in their storage layouts.
Input batches are distributed over `batch`; devices within each TP group share
the corresponding examples.

For each accumulation microbatch, forward and backward run under the model's
TP rules. When FSDP is enabled, the computation needs parameter data from other
batch replicas to form those TP slices. Backpropagation produces gradients,
and the accumulation buffer is constrained to the **parameter storage layout**.
Thus FSDP also affects how accumulated gradients are kept between microbatches.

Once accumulation finishes, the trainer constrains the gradients and parameters
to the **update layout**. The optimizer applies the update there, with its
partitioned state. Updated parameters are then constrained back to the
**storage layout**, ready for the next step. Without FSDP, this means making
updated parameter slices available to the other batch replicas again.

theseus expresses these requirements through logical-axis rules, concrete JAX
shardings, and sharding constraints. JAX compiles the communication needed to
satisfy them, including combining gradient contributions and gathering or
redistributing parameter slices. The layouts are the contract; this description
does not prescribe an exact collective schedule or promise that an entire model
is gathered at once. Validation uses the same computational TP rules without
performing an optimizer update.

## What gets checked before training

The trainer first constructs an abstract description of the training state:
array shapes and partition annotations, before materializing the full state.
It resolves the logical rules against the actual mesh and checks the storage
and update partitions. A tensor dimension split eight ways must divide evenly
by eight. Incompatible layouts fail with the tensor's state path, shape, and
partition specification rather than being silently padded.

For configuration, use `.shard(tp=2, fsdp=True)` on an execution chain or quick
session before job creation. The implementation is concentrated in
`theseus/base/axis.py` (policy and rules), `theseus/base/topology.py` (device mesh),
and `BaseTrainer` in `theseus/training/base.py` (layout transitions).
