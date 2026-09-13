from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Literal, Optional, Protocol, Sequence, cast, overload

import tiktoken

from theseus.config import configure, current_config, field
from theseus.data.datasets import ChatTemplate, ChatTurn


class Tokenizer(Protocol):
    @property
    def eot_token(self) -> int: ...

    def encode(self, text: str, allowed_special: Any = "all") -> list[int]: ...

    def encode_batch(
        self, texts: Sequence[str], allowed_special: Any = "all"
    ) -> list[list[int]]: ...

    def encode_ordinary(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def decode_batch(self, tokens_batch: Sequence[Sequence[int]]) -> list[str]: ...


@dataclass
class TokenizerConfig:
    backend: str = field("tokenizer/backend", default="tiktoken")
    name: str = field("tokenizer/name", default="cl100k_base")
    hf_use_fast: bool = field("tokenizer/huggingface/use_fast", default=True)
    hf_trust_remote_code: bool = field(
        "tokenizer/huggingface/use_remote_code", default=False
    )


class TikTokenTokenizer:
    def __init__(self, encoding: tiktoken.Encoding):
        self._encoding = encoding

    @property
    def eot_token(self) -> int:
        return int(getattr(self._encoding, "eot_token", 0))

    def encode(self, text: str, allowed_special: Any = "all") -> list[int]:
        if allowed_special is None:
            return cast(list[int], self._encoding.encode(text))
        return cast(
            list[int], self._encoding.encode(text, allowed_special=allowed_special)
        )

    def encode_batch(
        self, texts: Sequence[str], allowed_special: Any = "all"
    ) -> list[list[int]]:
        text_list = list(texts)
        if allowed_special is None:
            return cast(list[list[int]], self._encoding.encode_batch(text_list))
        return cast(
            list[list[int]],
            self._encoding.encode_batch(text_list, allowed_special=allowed_special),
        )

    def encode_ordinary(self, text: str) -> list[int]:
        return cast(list[int], self._encoding.encode_ordinary(text))

    def decode(self, tokens: Sequence[int]) -> str:
        return cast(str, self._encoding.decode(list(tokens)))

    def decode_batch(self, tokens_batch: Sequence[Sequence[int]]) -> list[str]:
        return cast(
            list[str],
            self._encoding.decode_batch([list(tokens) for tokens in tokens_batch]),
        )


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    @property
    def eot_token(self) -> int:
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is not None:
            return int(eos)
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is not None:
            return int(pad)
        return 0

    def encode(self, text: str, allowed_special: Any = "all") -> list[int]:
        del allowed_special
        return cast(list[int], self._tokenizer.encode(text, add_special_tokens=False))

    def encode_batch(
        self, texts: Sequence[str], allowed_special: Any = "all"
    ) -> list[list[int]]:
        del allowed_special
        tokenized = self._tokenizer(list(texts), add_special_tokens=False)
        return cast(list[list[int]], tokenized["input_ids"])

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return cast(
            str,
            self._tokenizer.decode(
                list(tokens),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )

    def decode_batch(self, tokens_batch: Sequence[Sequence[int]]) -> list[str]:
        return cast(
            list[str],
            self._tokenizer.batch_decode(
                [list(tokens) for tokens in tokens_batch],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )


class TrivialTokenizer:
    """Minimal tokenizer: space-splits text and casts each piece to int."""

    @property
    def eot_token(self) -> int:
        return 0

    def encode(self, text: str, allowed_special: Any = "all") -> list[int]:
        del allowed_special
        return [int(x) for x in text.split(" ")]

    def encode_batch(
        self, texts: Sequence[str], allowed_special: Any = "all"
    ) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encode(text)

    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(str(t) for t in tokens)

    def decode_batch(self, tokens_batch: Sequence[Sequence[int]]) -> list[str]:
        return [self.decode(tokens) for tokens in tokens_batch]


def _resolve_tokenizer_config(
    tokenizer_cfg: Optional[TokenizerConfig],
) -> TokenizerConfig:
    if tokenizer_cfg is not None:
        return tokenizer_cfg
    if current_config() is not None:
        return cast(TokenizerConfig, configure(TokenizerConfig))
    return TokenizerConfig()


def _build_chatml_tiktoken(name: str) -> tiktoken.Encoding:
    base = tiktoken.get_encoding(name)
    return tiktoken.Encoding(
        name=f"{name}_chatml",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={
            **base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
        },
    )


@lru_cache(maxsize=16)
def _build_tokenizer_cached(
    backend: str,
    name: str,
    hf_use_fast: bool,
    hf_trust_remote_code: bool,
) -> Tokenizer:
    backend_name = backend.lower().strip()

    if backend_name == "trivial":
        return TrivialTokenizer()

    if backend_name == "tiktoken":
        # Always use ChatML formatting for tiktoken backends.
        return TikTokenTokenizer(_build_chatml_tiktoken(name))

    if backend_name in {"huggingface", "hf", "transformers"}:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformers", "tokenizers"}:
                raise
            raise ModuleNotFoundError(
                "Tokenizer backend 'huggingface' requires transformers. "
                "Install with `uv sync --group huggingface` "
                "or `pip install 'libthx[huggingface]'`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=hf_use_fast,
            trust_remote_code=hf_trust_remote_code,
        )
        return HuggingFaceTokenizer(tokenizer)

    raise ValueError(
        f"Unknown tokenizer backend '{backend}'. Supported backends: "
        "'tiktoken', 'huggingface', 'trivial'."
    )


def get_tokenizer(tokenizer_cfg: Optional[TokenizerConfig] = None) -> Tokenizer:
    cfg = _resolve_tokenizer_config(tokenizer_cfg)
    return _build_tokenizer_cached(
        backend=cfg.backend,
        name=cfg.name,
        hf_use_fast=cfg.hf_use_fast,
        hf_trust_remote_code=cfg.hf_trust_remote_code,
    )


@overload
def encode_chat_template(
    template: ChatTemplate,
    encoder: Optional[Tokenizer] = None,
    system_prompt: Optional[str] = None,
    prompt: bool = False,
    *,
    tokenize: Literal[False],
) -> str: ...


@overload
def encode_chat_template(
    template: ChatTemplate,
    encoder: Tokenizer = ...,
    system_prompt: Optional[str] = None,
    prompt: bool = False,
    *,
    tokenize: Literal[True] = True,
) -> list[int]: ...


def encode_chat_template(
    template: ChatTemplate,
    encoder: Optional[Tokenizer] = None,
    system_prompt: Optional[str] = None,
    prompt: bool = False,
    *,
    tokenize: bool = True,
) -> list[int] | str:
    """
    Encode a chat template as tokens or formatted text.

    - For tiktoken, formatting is always ChatML.
    - For HuggingFace tokenizers, formatting uses tokenizer.apply_chat_template.

    Args:
        template: List of chat turns
        encoder: Tokenizer to use. Required when tokenize=True.
        system_prompt: Optional system prompt to prepend
        prompt: If True, append a generation prompt for autoregressive generation
        tokenize: If True, return token ids. If False, return formatted text.
    """
    # HuggingFace tokenizers should use their model-specific chat template.
    if isinstance(encoder, HuggingFaceTokenizer):
        turns: list[dict[str, str]] = []
        if system_prompt and system_prompt != "":
            turns.append({"role": "system", "content": system_prompt})
        for turn in template:
            turns.append({"role": turn.role, "content": turn.message})
        rendered = encoder._tokenizer.apply_chat_template(
            turns,
            tokenize=tokenize,
            add_generation_prompt=prompt,
            return_tensors=None,
        )
        if tokenize:
            ids_obj: Any = rendered

            # Some tokenizer versions return BatchEncoding with input_ids.
            if isinstance(rendered, dict) and "input_ids" in rendered:
                ids_obj = rendered["input_ids"]
            elif hasattr(rendered, "input_ids"):
                ids_obj = getattr(rendered, "input_ids")

            # Normalize tensor/ndarray-like values.
            if hasattr(ids_obj, "tolist"):
                ids_obj = ids_obj.tolist()

            if isinstance(ids_obj, tuple):
                ids_obj = list(ids_obj)

            if isinstance(ids_obj, list):
                if len(ids_obj) == 0:
                    return []
                if isinstance(ids_obj[0], list):
                    # apply_chat_template may return a batched shape; use first sample.
                    return [int(token) for token in ids_obj[0]]
                return [int(token) for token in ids_obj]

            raise TypeError(
                "Unexpected tokenized output type from apply_chat_template: "
                f"{type(ids_obj).__name__}"
            )
        return cast(str, rendered)

    # Build the full string first (ChatML).
    parts = []

    # Add system prompt if provided
    if system_prompt and system_prompt != "":
        parts.append("<|im_start|>system\n")
        parts.append(system_prompt)
        parts.append("<|im_end|>\n")

    # Add each turn
    for turn in template:
        parts.append(f"<|im_start|>{turn.role}\n")
        parts.append(turn.message)
        parts.append("<|im_end|>\n")

    # Add assistant prompt for autoregression
    if prompt:
        parts.append("<|im_start|>assistant\n")

    # Concatenate
    full_text = "".join(parts)

    if not tokenize:
        return full_text

    if encoder is None:
        raise ValueError(
            "encode_chat_template(tokenize=True) requires an encoder. "
            "Use tokenize=False to return formatted text."
        )

    tokens: list[int] = encoder.encode(full_text, allowed_special="all")
    return tokens


def encode_chat_template_with_mask(
    template: ChatTemplate,
    encoder: Tokenizer,
    system_prompt: Optional[str] = None,
) -> tuple[list[int], list[bool]]:
    """Encode a chat template and return a per-token assistant mask.

    Returns:
        (ids, assistant_mask): ids is the token list, assistant_mask[i] is True
        if token i belongs to an assistant turn (standard SFT masking).

    Uses incremental encoding: encodes progressively longer prefixes of the
    conversation to find exact token boundaries for each turn.
    """
    # Build the list of (turns_so_far, is_assistant_turn) pairs
    turns: list[dict[str, str]] = []
    if system_prompt and system_prompt != "":
        turns.append({"role": "system", "content": system_prompt})

    # Encode empty prefix to get the baseline length
    all_turns_with_role: list[tuple[dict[str, str], bool]] = []
    for turn in template:
        entry = {"role": turn.role, "content": turn.message}
        all_turns_with_role.append((entry, turn.role == "assistant"))

    if isinstance(encoder, HuggingFaceTokenizer):

        def _hf_encode_turns(turn_list: list[dict[str, str]]) -> list[int]:
            """Encode a turn list via apply_chat_template, always returning a flat id list."""
            result: Any = encoder._tokenizer.apply_chat_template(
                turn_list, tokenize=True, add_generation_prompt=False
            )
            # BatchEncoding / dict-like → extract input_ids
            if hasattr(result, "input_ids"):
                result = result.input_ids
            elif hasattr(result, "__getitem__") and not isinstance(
                result, (list, tuple)
            ):
                result = result["input_ids"]
            if hasattr(result, "tolist"):
                result = result.tolist()
            return [int(x) for x in result]

        # Incremental encoding: encode progressively longer prefixes
        # to find exact token boundaries per turn.
        current_turns = list(turns)  # starts with system prompt if any
        prev_len = 0

        if current_turns:
            prev_len = len(_hf_encode_turns(current_turns))

        assistant_mask: list[bool] = [False] * prev_len

        for entry, is_assistant in all_turns_with_role:
            current_turns.append(entry)
            ids_so_far = _hf_encode_turns(current_turns)
            new_len = len(ids_so_far)
            turn_token_count = new_len - prev_len
            assistant_mask.extend([is_assistant] * turn_token_count)
            prev_len = new_len

        all_ids: list[int] = ids_so_far
        return all_ids, assistant_mask

    # ChatML (tiktoken) path: build incrementally by encoding each segment
    assert encoder is not None

    all_ids_chatml: list[int] = []
    mask_chatml: list[bool] = []

    # System prompt
    if system_prompt and system_prompt != "":
        seg = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        seg_ids = encoder.encode(seg, allowed_special="all")
        all_ids_chatml.extend(seg_ids)
        mask_chatml.extend([False] * len(seg_ids))

    # Each turn
    for turn in template:
        seg = f"<|im_start|>{turn.role}\n{turn.message}<|im_end|>\n"
        seg_ids = encoder.encode(seg, allowed_special="all")
        is_assistant = turn.role == "assistant"
        all_ids_chatml.extend(seg_ids)
        mask_chatml.extend([is_assistant] * len(seg_ids))

    return all_ids_chatml, mask_chatml


def decode_chat_template(
    tokens: list[int] | str,
    encoder: Optional[Tokenizer] = None,
) -> ChatTemplate:
    """
    Decode tokens back into a ChatTemplate.

    Parses chatml format:
    <|im_start|>role
    message<|im_end|>

    Args:
        tokens: Token list or string (if encoder is None, treated as string)
        encoder: Tokenizer (if None, tokens is treated as the raw string)
    """
    # Decode tokens to text, or use directly if encoder is None
    if encoder is None:
        text = tokens if isinstance(tokens, str) else "".join(map(str, tokens))
    elif isinstance(tokens, str):
        text = tokens
    else:
        text = encoder.decode(tokens)

    # 1) Parse ChatML format first.
    chatml_matches = re.findall(
        r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>", text, re.DOTALL
    )
    if chatml_matches:
        return [
            ChatTurn(role=role.strip(), message=message.strip())
            for role, message in chatml_matches
        ]

    # 2) Parse Llama-3 style headers.
    # <|start_header_id|>user<|end_header_id|>\n\nmessage<|eot_id|>
    llama3_pattern = (
        r"<\|start_header_id\|>\s*([^<\n]+?)\s*<\|end_header_id\|>\s*"
        r"(.*?)(?=<\|eot_id\|>|<\|start_header_id\|>|$)"
    )
    llama3_matches = re.findall(llama3_pattern, text, re.DOTALL)
    if llama3_matches:
        return [
            ChatTurn(role=role.strip(), message=message.strip())
            for role, message in llama3_matches
            if message.strip() != ""
        ]

    # 3) Parse [INST]...[/INST] style (Llama-2 style).
    inst_matches = re.findall(
        r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)", text, re.DOTALL
    )
    if inst_matches:
        template: ChatTemplate = []
        for user_msg, assistant_msg in inst_matches:
            user_clean = user_msg.strip()
            if user_clean:
                template.append(ChatTurn(role="user", message=user_clean))
            assistant_clean = assistant_msg.strip()
            if assistant_clean:
                template.append(ChatTurn(role="assistant", message=assistant_clean))
        if template:
            return template

    # 4) Best-effort fallback: treat full text as assistant output.
    cleaned = text.strip()
    if cleaned == "":
        return []
    return [ChatTurn(role="assistant", message=cleaned)]
