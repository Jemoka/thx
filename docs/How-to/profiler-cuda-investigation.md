# CUDA failure during live profiling

Investigation: 2026-09-10–11. Status: reproduced; controlled thread comparison identifies concurrent profiling transitions as the actionable cause; proposed scheduling solution tested. The exact native failing instruction remains unidentified.

## Current conclusion

The reproduced failure is associated with starting/stopping profiling while
another thread continues submitting training work. The strongest comparison
used the **same JAX Python tracing calls**, model, and flags in both cases:
background-thread capture reproduced the CUDA error; training-thread capture
completed three captures and 10,000 steps. This rules out the RPC transport
being required for the failure. The native profiler lifecycle is the concern.

In the pinned XLA code, profiling shutdown finalizes CUPTI before its optional
synchronization helper. NVIDIA documents a crash risk if other threads continue
CUDA calls during finalization. This is the leading native explanation, supported
by source and the controlled comparison. It is not proof of which internal
allocation or GPU instruction failed. The earlier specific claim that a buffer
was freed while a kernel still used it was not established.

The proposed solution is to coordinate profiling transitions with training:
queue requests, finish pending work, start tracing, resume training, then hold
off new submissions and finish pending work before stopping tracing. This
retains collectives and command buffers. The prototype passed with command
buffers explicitly retained during profiling and with default flags. Further
comparisons passed even without explicit GPU waits, so waits alone are not
established as the essential ingredient; keeping the training thread from
submitting during start/stop is the shared behavior.

Keeping command buffers enabled during profiling **without coordination**
passed once and then crashed on repeat. Do not treat that flag as a sufficient
fix. The application scheduling proposal is tested on two RTX A5000 GPUs, not
validated across every GPU or production workload. Preserving arbitrary direct
JAX RPC requests requires a corresponding native lifecycle fix.

## Confirmed observations

A synthetic GPT trained across two NVIDIA RTX A5000 GPUs reproduced
`CUDA_ERROR_ILLEGAL_ADDRESS` when collecting a profile through the Theseus HTTP
capture endpoint. The endpoint invokes the JAX profiling RPC server, so this
exercises the same native profiling mechanism as connecting to that server directly.

The same model completed 10,000 steps without capture requests. PGLE remained
enabled in both experiments. This does not implicate ordinary PGLE operation.

| Setting | Value |
| --- | --- |
| Repository revision | `59dfe4de` |
| Host | `jagupard30`, selected through the configured `sc` cluster |
| GPUs | 2 × NVIDIA RTX A5000, Ampere |
| NVIDIA driver | `610.43.02` |
| JAX / jaxlib / CUDA plugin | `0.9.0.1` |
| CUDA dependency group | `cuda13`, frozen repository lockfile |
| CUPTI package | `nvidia-cuda-cupti==13.0.85` |
| NCCL package | `nvidia-nccl-cu13==2.28.9` |
| Optimization | `JAX_OPTIMIZATION_LEVEL=O2` |
| PGLE | enabled, three profiling runs |
| Parallelism | tensor parallelism of 2 |
| Model | 4 GPT layers, width 512, 4 heads, MLP width 2048 |
| Sequence / vocabulary | 128 / 1024 |
| Parameters / activations | float32 / bfloat16 |
| Batch | 1, synthetic repeating tokens |
| Capture duration | 2 seconds per request |

The compiled training executable contained 31 command buffers, along with
asynchronous collectives. Neither was disabled for the reproduction.

### Reproduction sequence

1. Wait until training passes step 4,000.
2. Request a capture: request observed at step 4,004; response received at step 5,666.
3. Let training continue, then request another capture at step 5,816.
4. The second capture began at 14:44:17.380 on the worker's clock.
5. CUPTI reported activity-buffer processing at 14:44:19.536.
6. At 14:44:19.823, recording a CUDA event reported the illegal-address error.
7. Training stopped advancing at approximately step 6,080.

The event-recording operation is where the asynchronous GPU error was observed;
it does not identify the GPU instruction that originally failed. The timing
places the observation just after the second capture's collection interval.

Earlier captures, at approximately steps 17 and 1,004, encountered a different
error: a checkpoint operation attempted PGLE profiling while the requested
profiling session was active. Those attempts are not counted as reproductions
of the CUDA error. Delaying capture separated that issue from the requested failure.

## Mechanism being tested

The pinned XLA source changes execution behavior when a profiler session is
active. It can execute the original operation sequence instead of a command
buffer. Async collective start/done pairs are kept together by the conversion
pass, so a simple split-pair explanation is not supported by this source.

There is also a concrete profiling-lifecycle concern: XLA's CUPTI shutdown path
disables tracing, flushes activity records, calls CUPTI finalization, and only
then calls its optional synchronization helper. That helper synchronizes
devices only when its corresponding option is enabled. Theseus permits the training thread
to continue submitting GPU work throughout the RPC capture's start and stop.

NVIDIA's CUPTI 13.0 documentation says that when finalization is not invoked
from a CUDA API exit callback, the client must arrange the required CUDA
synchronization and activity-buffer flushing beforehand. The observed error
timing is consistent with this concern, but timing and source inspection alone
do not prove that finalization is the exact cause.

NVIDIA also describes the concurrency problem directly in its
[dynamic profiling lifecycle documentation](https://docs.nvidia.com/cupti/13.0.0/main/main.html#dynamic-attach-and-detach):
other application threads can continue CUDA calls during finalization and
encounter partially dismantled CUPTI state. It requires coordination through a
CUDA API exit callback or prior synchronization and flushing. This supports a
specific shutdown-concurrency hypothesis, but does not identify the instruction
responsible for this reproduction's asynchronous GPU error.

The pinned XLA advanced profiling options parser does not expose its internal
`cupti_finalize` or `sync_devices_before_stop` fields as configurable options.
An invented Python profiler option cannot test these fields; it would be rejected
as unrecognized. See the
[pinned options parser](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/profiler/gpu/cupti_tracer_options_utils.cc).

### Why normal PGLE can work

JAX explicitly waits for an execution result before ending a PGLE profiling
session (`out[0].block_until_ready()` inside the PGLE context in `pxla.py`).
The remote timed capture does not have that training-step completion guarantee.
This is a concrete distinction between successful ordinary PGLE runs and a
capture requested while training is advancing asynchronously.

## Solution experiments

Two discriminating experiments were run:

* Keep command buffers enabled during profiling, leaving the asynchronous
  request mechanism otherwise unchanged. This tests whether XLA's automatic
  execution-mode change is necessary for the failure.
* Have a capture request wait for a training-step boundary; finish pending GPU
  work before starting capture, and again before stopping it. Keep command
  buffers and collectives enabled. This tests coordinated profiling transitions.

An alternative application-level proposal is to queue
capture requests and service their start/stop transitions through the training
loop. A multi-process implementation would need all workers to participate at
the same boundary. Normal training should not acquire a per-step GPU barrier;
the synchronization belongs only at requested profiling transitions.

### Coordinated capture with command buffers enabled

The coordinated-capture prototype completed three captures and all 10,000
training steps. Capture intervals ended at steps 4,107, 4,373, and 4,666.
The command-buffer profiling flag was explicitly enabled throughout this test;
PGLE and asynchronous collectives were retained.

| Capture | GPU devices represented | GPU duration events | Events with graph-related names |
| --- | ---: | ---: | ---: |
| 1 | 2 | 102,100 | 7,400 |
| 2 | 2 | 109,247 | 7,918 |
| 3 | 2 | 109,247 | 7,918 |

The traces were parsed as compressed Perfetto JSON; these are device events,
not merely host-side traces. The process subsequently spent substantial time in cleanup, then exited
normally with status 0.

### Original RPC capture with command buffers enabled

The comparison wrapper returned status 0. Its harness asserts that all three
capture requests succeeded, that training advanced across the captures, and
that training progressed afterward to the configured 10,000-step endpoint.
It used `--xla_enable_command_buffers_during_profiling=true` without the
coordinated-capture prototype.

The saved comparison log was subsequently retrieved. Capture responses arrived
at steps 5,516, 7,378, and 9,089; training reached step 10,000, cleanup completed,
and the wrapper reported status 0. The remote handler requires an exported
Perfetto trace before returning success. Unlike the coordinated test, the GPU
event contents of these three traces were not independently parsed.

### Follow-up: coordinated capture with default behavior

SLURM job `17368588` first ran the coordinated prototype with `XLA_FLAGS`
unset. The three capture intervals were steps 4,007–4,117, 4,266–4,356,
and 4,533–4,641. Training reached 10,000 steps, cleanup completed, and the
wrapper returned status 0. Thus the explicit command-buffer profiling flag
was not required for this successful coordinated run. GPU events in these
particular traces were not independently parsed.

### Follow-up: original RPC capture repeat failed

The same batch then repeated the original RPC case with command buffers
retained during profiling. Capture began at step 4,007. CUPTI reported activity
buffer processing at 20:42:12.786; CUDA event recording reported an illegal
address at 20:42:13.105. The request eventually returned at step 4,162, but
training had failed, and the process exited with status 134. A returned capture
response alone is therefore insufficient to validate a solution.

| Capture method | Command buffers during profiling | Outcome |
| --- | --- | --- |
| No requested capture | Default | 10,000 steps completed |
| Original RPC | Default | CUDA failure during second capture |
| Original RPC | Explicitly enabled | First run passed; repeat failed during first capture |
| Coordinated start/stop | Explicitly enabled | Three captures, 10,000 steps, exit 0 |
| Coordinated start/stop | Default | Three captures, 10,000 steps, exit 0 |
| Training-thread capture, no explicit waits | Default | Two runs each passed three captures and 10,000 steps |
| Training-thread capture, start-only or stop-only waits | Default | Each passed three captures and 10,000 steps |
| Background-thread Python capture, no explicit waits | Default | CUDA failure during first capture |

### Proposed solution and limits

Pursue coordinated start/stop as the candidate solution. Keep the existing
application request endpoint, queue capture requests, and service transitions
at training boundaries with pending GPU work completed. The tested prototype
uses `jax.block_until_ready(jax.live_arrays())` plus `jax.effects_barrier()`
before JAX `start_trace` and `stop_trace`. To retain command buffers during
capture as well as ordinary training, combine this coordination with
`--xla_enable_command_buffers_during_profiling=true`; that combination was
tested successfully, whereas the flag alone failed its repeat. A production implementation needs
explicit ownership of training work, request cancellation, error propagation,
and participation by all workers for multi-process training. A periodically
polled profiling-active flag cannot ensure this coordination.

Do not adopt the flag-only change as a fix. The failed repeat shows that
execution-mode switching alone cannot explain every observed failure.
The thread comparison below reproduces the failure with background capture
and passes with training-thread capture. It supports coordinating transitions
with submissions. The exact failing GPU operation and allocation remain
unidentified. Broader hardware and production validation would be needed for
a general reliability claim. The application patch described below is now implemented locally and remains uncommitted.

### Scope of the proposed repair

The current listener is created by `theseus/training/profiler.py`; bootstrap
sets optimization and PGLE defaults. The application can keep its HTTP capture
page while scheduling JAX tracing calls through `BaseTrainer.train()`.
An application-only scheduling change does **not** repair arbitrary direct
requests to the native JAX RPC port: those requests bypass the application
queue. Retaining that port with unchanged behavior would leave the original
failure path available.

A repair that preserves direct JAX RPC profiling therefore belongs in the
native profiling lifecycle. The candidate upstream change is to coordinate
profiling transitions with CUDA submissions and obey CUPTI's documented
finalization ordering. A synchronization call by itself is insufficient if
other threads can immediately enqueue more work. That native change has not
been implemented or tested here; the application prototype demonstrates a
possible scheduling approach, not a repaired JAX RPC server.

### Completed comparison of explicit waits

Job `17368706` tested the same coordinated Python capture implementation with
three wait modes: neither transition, start only, and stop only. Each case
runs in a fresh process with default XLA flags, PGLE enabled, and the same
model. A six-minute process limit bounds stalled cases. This comparison holds
the capture API and calling thread constant while varying explicit GPU waits.
The no-wait case completed three captures and all 10,000 steps with exit 0
(`boundary-none.log`, wrapper `boundaries-job.log`). This weakens any claim
that explicit GPU waits themselves are necessary: using the training thread
for synchronous start/stop also pauses new submissions from that thread.
The start-only case also completed three captures (responses at steps 4,074,
4,319, and 4,601), all 10,000 steps, and exited 0 (`boundary-start.log`).
The stop-only case completed three captures (responses at steps 4,079,
4,309, and 4,590), all 10,000 steps, and exited 0 (`boundary-stop.log`).
All three variants passed. These are single runs per variant, not a reliability
guarantee or proof that all CUDA work was idle. They do not show that explicit
waits are the essential difference from RPC capture.

### Isolating the calling thread

Job `17368747` compares the same Python tracing calls on a background thread
and the training thread, both without explicit GPU waits. This holds the API
constant while changing whether the training loop can submit more work during
start/stop. Both cases use default XLA flags and retain PGLE and collectives.
Both cases have finished. Background capture started at step 4,009 and
reported the CUDA illegal-address error at worker time 21:10:25.266, before
its first stop/export returned at step 4,132. Training then stopped advancing.
The process remained alive until its six-minute limit, so its wrapper status
is 124; the saved log independently confirms the CUDA failure.

The training-thread case completed captures at steps 4,110, 4,345, and 4,636,
then reached 10,000 steps, finished cleanup, and exited 0. This also repeats
the earlier successful no-wait training-thread case.

The batch itself completed with status 0 because it deliberately continued
after the first case to run the comparison. That batch status must not be
mistaken for both cases passing. `thread-job.log` records the individual
statuses (124 and 0); the detailed logs establish failure versus completion.

The pinned native Python binding releases the GIL during `stop_and_export`,
so a background Python call does not implicitly suspend the training thread
throughout trace shutdown. See the
[profiling binding](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/python/profiler.cc#L168-L183).
This makes the thread comparison useful for testing concurrent submission,
although it still cannot identify a failing GPU instruction by itself.

## Reproduction artifacts

The evidence filenames recorded during the investigation are listed below.
These temporary diagnostic scripts and logs are not included in the repository:

* `reproduction.log`: original illegal-address failure.
* `cooperative.log`: three coordinated captures and 10,000 completed steps.
* `comparison-result.log`: original RPC capture with command buffers retained,
  three successful requests, 10,000 completed steps, and status 0.
* `cooperative-default.log`: coordinated capture with default flags, exit 0.
* `remote-keepgraphs-repeat.log`: failed repeat with command buffers retained.
* `thread-background.log`: CUDA failure using Python tracing on a background thread.
* `thread-training.log`: matched training-thread case, three captures and 10,000 steps.
* `thread-job.log`: individual statuses for both thread cases.
* `model_thread_comparison.py`, `test-thread-comparison.sh`: matched comparison harness and batch script.
* `transition-job.log`: both follow-up process exit statuses.
* `boundary-none.log`, `boundary-start.log`, `boundary-stop.log`: completed
  wait variants; `boundaries-job.log` records all three exit statuses as 0.
* `model_repro.py` and `model_cooperative.py`: diagnostic harnesses derived from
  `scripts/smoke/profiler.py`. Later diagnostic revisions add progress messages
  and preserve output directories to avoid timing temporary-file cleanup.

These are investigation artifacts, not production profiler changes. Full
coordinated-capture traces are additionally archived locally outside the
repository in `/tmp/theseus-profiler-investigation/cooperative-evidence.tar.gz`.

The completed follow-up was SLURM job `17368588`. Its standalone batch script
copied both result logs to the login host; local copies are listed above.

The uv environment was successfully installed from the frozen lockfile in a
private, persistent node-local directory. Initial shared-cache locking and
shared-storage quota problems were setup issues; the completed experiments
used the working environment under `/scr/biggest/houjun/profiler-investigation`.

## Sources

The JAX 0.9.0.1 source pins XLA revision
`bb760b047bdbfeff962f0366ad5cc782c98657e0`:

* [JAX's XLA revision](https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/third_party/xla/revision.bzl)
* [Command-buffer execution and profiling fallback](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/gpu/runtime/command_buffer_thunk.cc)
* [Async-region conversion](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc)
* [JAX waits before ending PGLE profiling](https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/jax/_src/interpreters/pxla.py#L1352-L1374)
* [CUPTI shutdown ordering](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/profiler/gpu/cupti_tracer.cc)
* [NVIDIA CUPTI 13.0 finalization contract](https://docs.nvidia.com/cupti/13.0.0/api/group__CUPTI__ACTIVITY__API.html)

## One-step application patch

The agreed implementation replaces duration-based capture with a request for
one optimizer step. `Profiler.measure_step()` encloses the existing training
step annotation. It synchronizes before starting collection and before stopping
collection, only when a request exists. The HTTP worker receives the stopped
session and serialized data through a Future, then exports and prepares the
download. Normal steps never wait for the requester or trace export.

The JAX native listener and subprocess collector are removed. The HTTP page
now offers “Capture next training step”; the duration field is removed.
Capture currently requires one JAX process, which may use multiple GPUs.
Multi-process capture returns an explicit unsupported response rather than
attempting an uncoordinated partial capture.

The implementation uses the pinned private `jaxlib._profiler.ProfilerSession`
API because its `stop()` and `export()` operations separate collection from
export. JAX's public `stop_trace()` combines those operations. A captured step
still incurs synchronization and collection/serialization time on the training
thread; exporting and preparing the download run on the HTTP worker.

Job `17369570` passed all 38 focused HTTP/lifecycle/base-trainer tests. The
first two-GPU run completed 10,000 steps and exited 0 with command buffers
retained during profiling, PGLE enabled, and three requested captures. Every
downloaded trace contained exactly one training-step annotation, both GPUs,
and 1,006 GPU duration events. Responses arrived at steps 4,026, 4,227, and
4,285. The second run also completed 10,000 steps and exited 0; responses
arrived at steps 4,018, 4,188, and 4,365. All three repeat traces again contained
one training step, both GPUs, and 1,006 GPU duration events. Thus the actual
patch passed six one-step captures across two independent training processes.
Evidence is in `one-step-tests.log`, `one-step-job.log`, `one-step-1.log`, and
`one-step-2.log` under the local investigation directory. Ruff and the diff
whitespace check passed. No commit was made during validation.
The first test attempt exposed six trainer mocks needing a context manager;
those mocks were corrected before the successful 38-test run.

### Hunk review against the requested ground rules

* Profiler imports/state: replace subprocess plumbing with a standard Future;
  no helper hierarchy or additional conceptual module (rules 1, 4, 6, 7).
* Request handling/export: the existing HTTP action queues, awaits, and exports;
  native collection is separate, and unrelated training logic stays out
  (rules 2, 3, 4, 6).
* `measure_step`: one semantic context manager expresses the front/back
  lifetime, is reused for every optimizer step, and introduces no one-use
  production helpers (rules 1, 2, 3, 5, 6, 7).
* Listener/download/close: preserve the existing actions, remove the native
  listener, serve the new trace location, and release queued requests on close
  (rules 3, 4, 5, 7).
* Training hook: one context-manager expression beside the step annotation
  makes the capture boundary explicit (rules 1, 2, 6, 7).
* HTML and smoke request: remove the duration input and request parameter;
  maintain the existing page and smoke-test structure (rules 2, 4, 5, 7).
* Tests: shared request/queue fixtures exercise lifecycle and thread ownership;
  existing trainer mocks use standard `nullcontext`, with no helper framework
  (rules 1, 4, 5, 6, 7). Test callbacks implement the mocked interfaces.
* Investigation: verify the private API and test the actual patch before
  claiming success; report unresolved limitations explicitly (rule 8).

## Local XProf download

Each completed capture now offers both a Perfetto JSON download and an XProf
ZIP download. The ZIP contains the complete native `plugins/profile/...`
export, including `.xplane.pb` files, with paths relative to the capture root.
Extract the ZIP and run `xprof --logdir /path/to/extracted-folder` locally.
Archive creation runs on the HTTP worker after collection; the training
boundary and native-listener removal are unchanged.

Job `17369652` passed all 12 profiler tests in 8.45 seconds. Tests cover native
file bytes and archive layout, plus real JAX CPU captures downloaded through
HTTP and inspected as ZIP archives with nonempty `.xplane.pb` files. This is
an end-to-end capture/download test, not an automated XProf browser test.
The result was recorded as `xprof-tests.log`, a temporary diagnostic log.

Hunk review: the export action adds standard-library ZIP creation (rules 1, 2,
4, 7); the native route reuses the existing download handler through `partial`
(rules 1, 3, 5, 6, 7); the page adds a clearly named download and local usage
instructions (rules 4, 6, 7); existing tests verify the real download payload
without a new helper framework (rules 1, 4, 5, 8). No commit was made during validation.
