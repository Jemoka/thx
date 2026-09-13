"""GPU regression: two HTTP captures must leave the real trainer advancing.

Run on an otherwise idle GPU: uv run python -m scripts.smoke.profiler
The process exits with a stack dump if CUDA stalls. Artifacts are temporary.
"""

import faulthandler
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import urllib.request

import jax
from loguru import logger

from scripts.smoke.gpt_comet_fixture import TinyGPT
from theseus.quick import quick
from theseus.training.profiler import Profiler


if __name__ == "__main__":
    if jax.default_backend() != "gpu":
        raise RuntimeError("This regression requires a GPU with CUPTI available.")
    faulthandler.dump_traceback_later(180, exit=True)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    captures: list[int] = []
    errors: list[Exception] = []
    with TemporaryDirectory() as root, quick(Path(root)) as q:
        q.build(TinyGPT, "profiler-smoke")
        c = q.config
        c.architecture.n_layers = 4
        c.architecture.n_embd = 128
        c.architecture.n_head = 4
        c.architecture.intermediate_size = 512
        c.architecture.block_size = 128
        c.architecture.vocab_size = 1024
        c.architecture.dtype.param = "float32"
        c.architecture.dtype.activation = "bfloat16"
        c.training.batch_size = 1
        c.training.per_device_batch_size = 1
        c.training.tokens = 3_840_000
        c.training.validation = False
        c.training.evaluate = False
        c.training.analyze = False
        c.logging.remote = False
        c.logging.report_interval = 1000
        c.logging.checkpoint_interval = 100_000
        job = q.create()
        job._profiler = Profiler(0, node=job.node)
        job._profiler.start()
        url = f"http://127.0.0.1:{job._profiler.port}/profile"

        def capture() -> None:
            try:
                while job.node.seq < 10:
                    time.sleep(0.1)
                for _ in range(2):
                    before = job.node.seq
                    with urllib.request.urlopen(url, b"", timeout=100) as response:
                        result = json.load(response)
                    assert result["trace"].startswith("/trace/capture-")
                    assert job.node.seq > before
                    captures.append(job.node.seq)
                    time.sleep(2)
            except Exception as error:
                errors.append(error)

        worker = Thread(target=capture, daemon=True)
        worker.start()
        q()
        worker.join(timeout=65)
        assert not worker.is_alive(), "Capture did not finish"
        assert not errors, errors
        assert len(captures) == 2, captures
        assert job.node.seq > captures[-1], "Training stopped after capture"
        print(json.dumps({"captured_steps": captures, "final_step": job.node.seq}))
    faulthandler.cancel_dump_traceback_later()
