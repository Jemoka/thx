# Extending Components

???+ warning "Coming Soon"
    Heyoooo so I ran out of time writing docs as I have to do actual machine learning. I'll eventually catch this up but as of right now here's my friend `gpt-6-astra` who will do the talking.

---

Choose the class closest to the behavior you want to change. These guides cover
base classes; the API reference lists their concrete implementations.

## Trainer Extension

[Extend a trainer](trainer.md) to choose its model, data, optimizer, schedule,
evaluations, and analyses, or change its forward computation.

## Extension Points

| Change | Inherit from | Guide |
|---|---|---|
| Define a configurable model building block | `Module` | [Module contract](module.md) |
| Change embeddings, layer execution, logits, or loss | `GPT` | [GPT](gpt.md) |
| Change sublayers and residual connections | `Block` | [Transformer blocks](block.md) |
| Change Q/K/V projections, masks, or attention computation | `SelfAttention` | [Attention](attention.md) |
| Change the feed-forward computation | `MLP` | [MLP](mlp.md) |
| Change expert selection or expert implementations | `MoE` | [Mixture of experts](moe.md) |

## Reusable Modules

Use these inside your implementation. Links go directly to the API reference.

| Building block | Reference |
|---|---|
| Rotary position encoding | [RotaryPosEncoding](../../../Reference/model/layers/rope.md) |
| Multimodal rotary position encoding | [MRotaryPosEncoding](../../../Reference/model/layers/mrope.md) |
| Layer normalization | [LayerNorm](../../../Reference/model/layers/layernorm.md) |
| RMS normalization, gated RMS normalization, and `rms()` | [RMSNorm and helpers](../../../Reference/model/layers/rmsnorm.md) |
| SwiGLU activation | [swiglu](../../../Reference/model/activations/swiglu.md) |
