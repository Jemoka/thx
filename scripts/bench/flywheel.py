"""Compare exact loader output and timing against a git revision.

Local fixture:
    python -m scripts.bench.flywheel --root /tmp/flywheel --steps 256

Existing (read-only) dataset:
    python -m scripts.bench.flywheel --root /mnt/data --existing mnli \
        --styles padded --block-size 512 --steps 8

Trials alternate versions. Caches are OS/filesystem managed, never evicted;
first-touch and warmed timings must not be interpreted as cold-cache controls.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from theseus.base import Node


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/flywheel"))
    parser.add_argument("--baseline", default="d34ae37")
    parser.add_argument(
        "--existing", help="Dataset directory under root; never written"
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=["pmd", "padded", "contrastive"],
        default=["pmd", "padded", "contrastive"],
    )
    parser.add_argument("--mode", choices=["reader", "async"], default="reader")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--after-first", action="store_true")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/flywheel-report.json")
    )
    args = parser.parse_args()
    if args.existing and len(args.styles) != 1:
        parser.error("--existing requires exactly one --styles entry")
    if min(args.steps, args.trials, args.batch_size, args.block_size, args.rows) < 1:
        parser.error("sizes, steps, and trials must be positive")
    if not args.existing:
        rows, width = args.rows, args.block_size + 1
        for style in args.styles:
            path = args.root / style
            path.mkdir(parents=True, exist_ok=True)
            sides = ["pos", "neg"] if style == "contrastive" else [None]
            for side in sides:
                filename = f"train.{side}.bin" if side else "train.bin"
                file = path / filename
                if file.exists() and file.stat().st_size != rows * width * 4:
                    parser.error(f"Existing fixture has a different shape: {file}")
                if not file.exists():
                    data = np.memmap(
                        file, mode="w+", dtype=np.uint32, shape=(rows, width)
                    )
                    for start in range(0, rows, 4096):
                        stop = min(start + 4096, rows)
                        data[start:stop] = (
                            np.arange(width, dtype=np.uint32)[None, :]
                            + np.arange(start, stop, dtype=np.uint32)[:, None]
                            + 1
                        )
                    data.flush()
                    del data
                if style != "pmd" and not (path / f"{filename}.mask").exists():
                    mask = np.memmap(
                        path / f"{filename}.mask",
                        mode="w+",
                        dtype=np.bool_,
                        shape=(rows, width),
                    )
                    mask[:] = True
                    mask[:, : width // 4] = False
                    mask.flush()
                    del mask
            if style != "pmd":
                shape = [rows, width]
                if style == "contrastive":
                    shape = {"pos": shape, "neg": shape}
                (path / "shape.json").write_text(json.dumps({"train": shape}))

    # Isolate the entire baseline pipeline, including planning and validation.
    package = "theseus.training._flywheel_baseline"
    module = types.ModuleType(package)
    module.__path__ = []
    sys.modules[package] = module
    before, after = {}, {}
    for name in ("stream", "dataset", "pmd", "padded", "contrastive"):
        source = subprocess.check_output(
            ["git", "show", f"{args.baseline}:theseus/training/flywheel/{name}.py"],
            text=True,
        ).replace("theseus.training.flywheel", package)
        module = types.ModuleType(f"{package}.{name}")
        sys.modules[module.__name__] = module
        exec(compile(source, f"{args.baseline}/{name}.py", "exec"), module.__dict__)
        before[name] = module
        after[name] = importlib.import_module(f"theseus.training.flywheel.{name}")

    spec = SimpleNamespace(
        hardware=SimpleNamespace(
            hosts=[SimpleNamespace(cluster=SimpleNamespace(data_dir=args.root))]
        )
    )
    results, digests = [], {}
    for style in args.styles:
        cls_name = {
            "pmd": "MemmapDataset",
            "padded": "PaddedDataset",
            "contrastive": "ContrastivePaddedDataset",
        }[style]
        for trial in range(args.trials):
            versions = (
                ("before", "after")
                if (trial + args.after_first) % 2 == 0
                else ("after", "before")
            )
            for version in versions:
                modules = before if version == "before" else after
                begin = perf_counter()
                reader = getattr(modules[style], cls_name)(
                    spec, args.block_size, args.existing or style
                )
                node = Node(name="benchmark", seq=args.start)
                if args.mode == "async":
                    loader = modules["stream"].AsyncStrategy(
                        [reader], [1.0], args.batch_size, node=node, seed=args.seed
                    )
                else:
                    loader = modules["stream"].batches(
                        [reader],
                        [1.0],
                        args.batch_size,
                        seed=args.seed,
                        start=args.start,
                    )
                setup = perf_counter() - begin
                digest, times = hashlib.sha256(), []
                loop_start = perf_counter()
                try:
                    for step in range(args.steps):
                        begin = perf_counter()
                        batch = (
                            loader.get_batch() if args.mode == "async" else next(loader)
                        )
                        times.append(perf_counter() - begin)
                        for key in sorted(batch):
                            digest.update(key.encode())
                            digest.update(batch[key].dtype.str.encode())
                            digest.update(batch[key].tobytes())
                        if args.mode == "async":
                            node.update(node.next())
                finally:
                    loader.close()
                elapsed = perf_counter() - loop_start
                value = digest.hexdigest()
                if style in digests and value != digests[style]:
                    raise AssertionError(f"Output changed for {style}: {version}")
                digests[style] = value
                result = dict(
                    style=style,
                    version=version,
                    trial=trial,
                    setup_s=setup,
                    first_batch_s=times[0],
                    total_s=elapsed if args.mode == "async" else sum(times),
                    batches_per_s=len(times)
                    / (elapsed if args.mode == "async" else sum(times)),
                    timing=(
                        "wall time including output verification"
                        if args.mode == "async"
                        else "time inside reader calls"
                    ),
                    seconds=times,
                    sha256=value,
                )
                results.append(result)
                args.report.write_text(
                    json.dumps(
                        {
                            "arguments": vars(args),
                            "cache": "OS-managed; no forced eviction",
                            "results": results,
                        },
                        default=str,
                        indent=2,
                    )
                )
                print(json.dumps(result), flush=True)
                del loader, reader
