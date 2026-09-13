"""Probe padded random I/O on disjoint file stripes, checking exact output.

Each version first touches a different stripe, in alternating order. This avoids
deliberately warming the same rows for its competitor, but does not evict any OS
or filesystem caches. Use an otherwise untouched dataset for useful measurements.
"""

import argparse
import json
import subprocess
import types
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from theseus.training.flywheel.padded import PaddedDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", default="d34ae37")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/flywheel-storage.json")
    )
    args = parser.parse_args()
    if min(args.block_size, args.batch_size, args.trials) < 1:
        parser.error("sizes and trials must be positive")
    module = types.ModuleType("baseline_padded")
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:theseus/training/flywheel/padded.py"],
        text=True,
    )
    exec(compile(source, "baseline/padded.py", "exec"), module.__dict__)
    spec = SimpleNamespace(
        hardware=SimpleNamespace(
            hosts=[SimpleNamespace(cluster=SimpleNamespace(data_dir=args.root))]
        )
    )
    before = module.PaddedDataset(spec, args.block_size, args.dataset)
    after = PaddedDataset(spec, args.block_size, args.dataset)
    size = before._size("train")
    stripe = size // (args.trials * 2)
    if stripe < 1:
        parser.error("not enough rows for disjoint stripes")
    results = []
    for trial in range(args.trials):
        order = ("before", "after") if trial % 2 == 0 else ("after", "before")
        for version in order:
            number = len(results)
            indices = np.random.default_rng(51 + number).integers(
                stripe, size=args.batch_size
            )
            indices += number * stripe
            reader = before if version == "before" else after
            parallel = version == "after" and reader._reader._parallel
            start = perf_counter()
            actual = reader._read_rows(indices, "train")
            seconds = perf_counter() - start
            result = dict(
                trial=trial,
                version=version,
                parallel=parallel,
                seconds=seconds,
                rows=indices.tolist(),
            )
            print(json.dumps(result), flush=True)
            # Verify outside the measured interval, after the first-touch read.
            reference = after if version == "before" else before
            expected = reference._read_rows(indices, "train")
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
                assert actual[key].dtype == expected[key].dtype
            result["exact_match"] = True
            results.append(result)
            args.report.write_text(
                json.dumps(
                    {
                        "arguments": vars(args),
                        "cache": "disjoint stripes; OS-managed, no eviction",
                        "results": results,
                    },
                    default=str,
                    indent=2,
                )
            )
