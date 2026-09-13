from dataclasses import dataclass

from datasets import load_dataset

from theseus.config import configure, field
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


@dataclass
class HarmfulQAConfig:
    conversation_type: str = field("data/harmfulqa/conversation_type", default="blue")
    split: str = field("data/harmfulqa/split", default="train")


@dataset("harmfulqa")
class HarmfulQA(ChatTemplateDataset):
    """HarmfulQA (Bhardwaj & Poria, 2023).

    Multi-round conversations labeled as safe ("blue") or harmful ("red").
    The ``config`` parameter selects the conversation type:
      - ``"blue"`` (default): safe conversations
      - ``"red"``: harmful conversations
      - ``"all"``: both types interleaved
    """

    CONFIG = HarmfulQAConfig

    def __init__(self) -> None:
        args = configure(self.CONFIG)
        conversation_type = args.conversation_type
        self.conversation_type = conversation_type
        ds = load_dataset("declare-lab/HarmfulQA", split=args.split)
        self._items: list[ChatTemplate] = []

        for row in ds:
            convs_key = f"{conversation_type}_conversations"
            if conversation_type == "all":
                for key in ["blue_conversations", "red_conversations"]:
                    self._extract_conversations(row, key)
            else:
                self._extract_conversations(row, convs_key)

    def _extract_conversations(self, row: dict, key: str) -> None:  # type: ignore[type-arg]
        convs = row.get(key, {})
        if not isinstance(convs, dict):
            return
        for conv_list in convs.values():
            if not isinstance(conv_list, list):
                continue
            turns: ChatTemplate = []
            for turn in conv_list:
                if not isinstance(turn, dict):
                    continue
                role = "user" if turn.get("from") == "human" else "assistant"
                message = turn.get("value", "")
                if message:
                    turns.append(ChatTurn(role=role, message=message))
            if turns:
                self._items.append(turns)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> ChatTemplate:
        return self._items[idx]
