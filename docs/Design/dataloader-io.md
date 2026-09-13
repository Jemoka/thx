# Dataloader I/O

The loader preserves sample plans, RNG consumption, batch order, host slicing,
mixture membership, masks, dtypes, and checkpoint replay. Physical reads may be
reordered; returned samples are restored to their original positions.

## Reading strategy

PMD still samples a size-weighted contiguous window of at most 256 MiB. It now
allocates that buffer without reading it all immediately. Only 4 MiB blocks
touched by requested samples and their shifted labels are filled, using a
process-wide pool of eight I/O workers. Loaded blocks stay in the window cache.
A cached strided view replaces per-sample Python slicing and stacking. The
window, its tail, and the permutation are unchanged.

Padded and contrastive readers start with direct mapped indexing. After a slow
read (over 10 ms), they try parallel positional reads. Sorted rows are grouped
into bounded reads, coalescing gaps up to 64 KiB and spans up to 4 MiB, then
scattered back into caller order. Token and mask files share the worker pool.
The reader retains parallel I/O only while its observed per-row cost beats
serial I/O, with a probe every 32 reads to accommodate changing cache conditions.
Fast reads return to direct indexing. Timing affects scheduling only.

Mappings and positional reads share open files, avoiding additional remote
opens and keeping both paths on the same file. Short reads are completed or
raise an explicit EOF error. All submitted reads finish before an I/O error is
propagated. File handles have the lifetime of the reader cache.

Causal padding allocates the final arrays directly when left padding is needed.
Contrastive masks avoid redundant copies. Empty-row validation avoids
token-sized comparison temporaries and groups replacement reads by PMD window
(or into one padded read), retaining each row's original retry seed and limit.
Worker copies own their scheduling state and PMD window caches. Existing sample
planning and the eight-batch async prefetch queue are unchanged.

## Measurements

Measured on macOS, Python 3.11.14, NumPy 2.4.2, against d34ae37
(the feat/timetravel base), on 2026-09-07. These are individual-machine results,
not universal throughput guarantees.

The user's `juice` alias mounts **JuiceFS backed by remote metadata and object
storage**, rather than a kernel NFS mount. The same volume was mounted separately
at `/tmp/theseus-dl-juice`, read-only, with background jobs disabled and a
512 MiB disk-cache limit. The original `~/theseus` directory was left alone.

Observed metadata ping latency was about 150 ms. Sequential 64 MiB probes took
18.4–23.8 seconds across mapped, positional, and threaded copies of separate
regions. Parallelism alone did not improve sustained transfer rate. An initial
PMD buffer fill therefore represents substantial startup latency.

### PMD

On the 326,119,742,308-byte FineWeb training file, batch size 1, context 1024,
seed 1764:

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| First batch | 51.74 s | 3.05 s |
| Bytes needed for first buffer fill | 256 MiB + 4 B | 4 MiB |
| Second batch | effectively cached | 2.71 s for another 4 MiB block |

The optimized version ran first; its two reads warmed 8 MiB for the baseline.
The outputs have the same SHA-256 digest. This demonstrates reduced work and
startup latency, not a 17× improvement in sustained whole-window throughput.
Long runs eventually consume the same window; large batches can touch most of
it immediately. The buffer still reserves up to 256 MiB plus the shifted token.

### Local cached throughput

Three alternating trials, 16,384 rows, stored width 1025, context 1024, 32 rows
per batch, 1,024 batches per trial. Median rates measure time inside synchronous
reader calls; checksum verification occurs outside those calls.

| Reader | Baseline batches/s | Optimized batches/s |
| --- | ---: | ---: |
| PMD | 16,167 | 21,509 |
| Padded | 15,096 | 15,398 |
| Contrastive | 9,799 | 9,642 |

PMD improved about 33%; the padded regimes were approximately unchanged.
Every trial's full output digest matches the baseline. Async comparisons also
verify output while starting at sequence 2045 and crossing the next plan
boundary. Async rates include consumer checksum time, since hashing can overlap
prefetch; summing queue wait times would exaggerate throughput.

### Padded remote reads

The mount was highly variable. Initial 32-row MNLI batches took 55–97 seconds
with the baseline. Subsequent disjoint-stripe probes with 16 rows produced
overlapping timing ranges for serial and parallel reads; parallel I/O was not
consistently faster. This motivated measuring both modes in the reader rather
than always using a thread pool.

The final adaptive reader processed eight 8-row MNLI batches in 42.97 seconds
(first batch 13.84 seconds). Reading those exact batches afterward with the
baseline took 3.11 seconds, with all but the first batch already cached. Their
digests match. That comparison is a correctness/cache-sensitivity observation,
**not a padded speedup claim**. No global caches were evicted; file-stripe
comparisons also cannot guarantee that backend blocks are cold.

## Reproduction and checks

[Timing summaries and output digests](dataloader-io-results.json) retain each
trial's measurements; large per-batch timing arrays are summarized.

`scripts/bench/flywheel.py` loads the baseline's entire planning/reading pipeline
under an isolated module name, checks output digests, and writes JSON timing
reports. Existing remote datasets are never modified.

```sh
python -m scripts.bench.flywheel --root /tmp/flywheel-local \
  --rows 16384 --steps 1024 --trials 3 --report /tmp/flywheel-local.json

python -m scripts.bench.flywheel --root /tmp/flywheel-local \
  --rows 16384 --steps 1024 --trials 3 --mode async --start 2045 \
  --report /tmp/flywheel-async.json

python -m scripts.bench.flywheel --root /tmp/theseus-dl-juice/data \
  --existing fineweb --styles pmd --block-size 1024 --steps 2 \
  --batch-size 1 --trials 1 --after-first --seed 1764 \
  --report /tmp/flywheel-juice-pmd.json

python -m scripts.bench.flywheel --root /tmp/theseus-dl-juice/data \
  --existing mnli --styles padded --block-size 512 --steps 8 \
  --batch-size 8 --trials 1 --after-first --seed 842 \
  --report /tmp/flywheel-juice-padded.json

python -m scripts.bench.flywheel_storage --root /tmp/theseus-dl-juice/data \
  --dataset mnli_llama --batch-size 16
```

The storage probe uses disjoint row stripes in alternating version order and
checks every result against its counterpart outside the measured interval.
Choose an otherwise untouched dataset to reduce cache contamination.

Validation: 41 tests across `test_flywheel_reads.py`,
`test_flywheel_determinism.py`, `test_contrastive_roundtrip.py`, and
`test_trainer_lifecycle.py`; Ruff; and mypy on the complete flywheel package.
Tests include independent rowwise retry equivalence for all three styles,
real optimizer/checkpoint replay, host topology changes, duplicate and tail
rows, samples spanning storage blocks, validation fallback, differing
contrastive widths, left padding, short reads, and EOF propagation.
