"""
Chip information.
"""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field


class TheoreticalFLOPS(BaseModel):
    """Theoretical FLOP/s, recorded in float TFLOP/s.

    Dense peaks per physical chip: BF16 matrix operations with FP32 accumulation,
    and native FP32 arithmetic (not TF32). None means unknown or unpublished.
    """

    model_config = ConfigDict(frozen=True)

    bfloat16: float | None = None
    float32: float | None = None


class Chip(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(description="name of hardware")]
    display_name: Annotated[str, Field(description="display name for logs")]
    memory: Annotated[int, Field(description="bytes of memory per chip")]
    flops: TheoreticalFLOPS = Field(default_factory=TheoreticalFLOPS)


SUPPORTED_CHIPS: dict[str, Chip] = {
    # Generic CPU: no specific microarchitecture or core count.
    "cpu": Chip(
        name="cpu",
        display_name="CPU",
        memory=int(32 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=None, float32=None),
    ),
    # GB10: published FP4 AI TOPS do not establish BF16/FP32 peaks.
    "gb10": Chip(
        name="gb10",
        display_name="Nvidia GB10",
        memory=int(64 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=None, float32=None),
    ),
    # HGX B200: 36 sparse BF16 PFLOP/s / 2 / 8 GPUs; 600 FP32 / 8.
    # https://www.nvidia.com/en-us/data-center/hgx/
    "b200": Chip(
        name="b200",
        display_name="Nvidia B200",
        memory=int(192 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=2250.0, float32=75.0),
    ),
    # SXM; published BF16 1,979 includes sparsity, so divide by two.
    # https://www.nvidia.com/en-us/data-center/h200/
    "h200": Chip(
        name="h200",
        display_name="Nvidia H200",
        memory=int(143.8 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=989.5, float32=67.0),
    ),
    # SXM; published BF16 1,979 includes sparsity, so divide by two.
    # https://www.nvidia.com/en-us/data-center/h100/
    "h100": Chip(
        name="h100",
        display_name="Nvidia H100",
        memory=int(80 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=989.5, float32=67.0),
    ),
    # Dense A100 peaks apply to both PCIe and SXM variants.
    # https://www.nvidia.com/en-us/data-center/a100/
    "a100-sxm4-80gb": Chip(
        name="a100-sxm4-80gb",
        display_name="Nvidia A100 SXM4 80GB",
        memory=int(80 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=312.0, float32=19.5),
    ),
    "a100": Chip(
        name="a100-sxm4-80gb",
        display_name="Nvidia A100 SXM4 80GB",
        memory=int(80 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=312.0, float32=19.5),
    ),
    "a100-pcie-40gb": Chip(
        name="a100-pcie-40gb",
        display_name="Nvidia A100 PCIe 40GB",
        memory=int(40 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=312.0, float32=19.5),
    ),
    # Server Edition estimate: 188 SMs * 1,024 dense BF16 FLOPs/clock * 2.430 GHz.
    # Architecture rate from Table 4; clock from the deployed Server cards.
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf
    # Inferred peak, not measured throughput; this shared key also matches
    # Workstation variants with different clocks (not distinguished here).
    "b6000": Chip(
        name="b6000",
        display_name="Nvidia RTX Pro 6000",
        memory=int(96 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=467.80416, float32=120.0),
    ),
    # RTX A6000 and RTX 6000 Ada: table 2, dense BF16 with FP32 accumulate.
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "a6000": Chip(
        name="a6000",
        display_name="Nvidia RTX A6000",
        memory=int(48 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=154.8, float32=38.7),
    ),
    "ada6000": Chip(
        name="ada6000",
        display_name="Nvidia RTX 6000 Ada",
        memory=int(48 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=364.2, float32=91.1),
    ),
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf
    "l40": Chip(
        name="l40",
        display_name="Nvidia L40",
        memory=int(48 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=181.05, float32=90.5),
    ),
    # Use the explicitly listed dense BF16 figure from the specifications table.
    # https://www.nvidia.com/en-us/data-center/l40s/
    "l40s": Chip(
        name="l40s",
        display_name="Nvidia L40S",
        memory=int(48 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=362.05, float32=91.6),
    ),
    # TPUs
    # 180 TFLOP/s per four-chip board; FP32 peak is not published.
    # https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/google-cloud-offer-tpus-machine-learning/
    "tpu-v2": Chip(
        name="tpu-v2",
        display_name="Google TPU v2",
        memory=int(8 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=45.0, float32=None),
    ),
    # https://docs.cloud.google.com/tpu/docs/v3
    "tpu-v3": Chip(
        name="tpu-v3",
        display_name="Google TPU v3",
        memory=int(16 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=123.0, float32=None),
    ),
    # https://docs.cloud.google.com/tpu/docs/v4
    "tpu-v4": Chip(
        name="tpu-v4",
        display_name="Google TPU v4",
        memory=int(32 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=275.0, float32=None),
    ),
    # https://docs.cloud.google.com/tpu/docs/v5e
    "tpu-v5e": Chip(
        name="tpu-v5e",
        display_name="Google TPU v5e",
        memory=int(16 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=197.0, float32=None),
    ),
    # https://docs.cloud.google.com/tpu/docs/v5p
    "tpu-v5p": Chip(
        name="tpu-v5p",
        display_name="Google TPU v5p",
        memory=int(95 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=459.0, float32=None),
    ),
    # PG199 board identifier: no verified precision-specific peak.
    "drive-pg199": Chip(
        name="drive-pg199",
        display_name="Nvidia Drive PG199",
        memory=int(32 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=None, float32=None),
    ),
    # Table 3: BF16 with FP32 accumulation (not FP16 accumulation).
    # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
    "rtx5090": Chip(
        name="rtx5090",
        display_name="Nvidia RTX 5090",
        memory=int(32 * 1024**3),
        flops=TheoreticalFLOPS(bfloat16=209.5, float32=104.8),
    ),
}
