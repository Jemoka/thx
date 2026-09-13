# Available chips

These are all accepted string keys in `SUPPORTED_CHIPS` in
`theseus/base/chip.py`. Use them in `.chip("h100")`, the CLI's `--chip` option,
or a dispatch inventory's `chips` mapping.

Table includes the numbers theseus uses for MFU calculations.

| String key | Chip | Memory (GiB) | BF16 (TFLOP/s) | FP32 (TFLOP/s) |
| --- | --- | ---: | ---: | ---: |
| `cpu` | CPU | 32 | — | — |
| `gb10` | Nvidia GB10 | 64 | — | — |
| `b200` | Nvidia B200 | 192 | 2250 | 75 |
| `h200` | Nvidia H200 | 143.8 | 989.5 | 67 |
| `h100` | Nvidia H100 | 80 | 989.5 | 67 |
| `a100-sxm4-80gb` | Nvidia A100 SXM4 80GB | 80 | 312 | 19.5 |
| `a100` | Nvidia A100 SXM4 80GB (alias) | 80 | 312 | 19.5 |
| `a100-pcie-40gb` | Nvidia A100 PCIe 40GB | 40 | 312 | 19.5 |
| `a6000` | Nvidia RTX A6000 | 48 | 154.8 | 38.7 |
| `ada6000` | Nvidia RTX 6000 Ada | 48 | 364.2 | 91.1 |
| `l40` | Nvidia L40 | 48 | 181.05 | 90.5 |
| `l40s` | Nvidia L40S | 48 | 362.05 | 91.6 |
| `tpu-v2` | Google TPU v2 | 8 | 45 | — |
| `tpu-v3` | Google TPU v3 | 16 | 123 | — |
| `tpu-v4` | Google TPU v4 | 32 | 275 | — |
| `tpu-v5e` | Google TPU v5e | 16 | 197 | — |
| `tpu-v5p` | Google TPU v5p | 95 | 459 | — |
| `drive-pg199` | Nvidia Drive PG199 | 32 | — | — |
| `rtx5090` | Nvidia RTX 5090 | 32 | 209.5 | 104.8 |

`a100` resolves to the canonical name `a100-sxm4-80gb`.
