---
template: home.html
hide:
  - navigation
  - toc
---

# theseus<img src="assets/colophon.png" alt="" class="theseus-colophon">

have you ever wanted to train a language model from scratch but hate writing boilerplate? previously, the solution to this was to work at a frontier lab with Research Engineers™.

now the solution is to make Jack™ (and also a cast of frontier coding models) do your research engineering. it will probably break a lot but what the heck at least i tried.

```python
from theseus.quick import quick

with quick("./results") as q:
    q.build("gpt/train/pretrain", "my-training-job")
    q.config.architecture.n_embd = 128
    q.config.architecture.n_layers = 4
    q.config.training.per_device_batch_size = 4

    job = q.create()
    job()
```

[Quickly start quickstart →](quickstart.md){ .md-button }
