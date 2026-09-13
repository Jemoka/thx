from datasets import load_dataset
from theseus.data.datasets import ChatTemplate, ChatTemplateDataset, ChatTurn
from theseus.registry import dataset


def template(
    context: str, question: str, choices: dict[str, str], answer: str
) -> ChatTemplate:
    choices_text = "\n".join(f"{k}: {v}" for k, v in choices.items())
    return [
        ChatTurn(
            role="user",
            message=f"""Read the following context and answer the question by selecting A, B, C, or D.

Context:
{context}

Question: {question}

{choices_text}

Answer with only the letter (A, B, C, or D):""",
        ),
        ChatTurn(role="assistant", message=answer),
    ]


@dataset("longbench")
class LongBench(ChatTemplateDataset):
    def __init__(self) -> None:
        self.ds = load_dataset("THUDM/LongBench-v2", split="train")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> ChatTemplate:
        item = self.ds[idx]
        choices = {
            "A": item["choice_A"],
            "B": item["choice_B"],
            "C": item["choice_C"],
            "D": item["choice_D"],
        }
        return template(item["context"], item["question"], choices, item["answer"])
