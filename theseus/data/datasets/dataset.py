from typing import Any, ClassVar, Generic, TypeVar, Iterator, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field

from theseus.config import field
from theseus.data.style import DatasetStyle


C = TypeVar("C")


@dataclass
class ChatTurn:
    role: str
    message: str
    metadata: dict[str, str] = dataclass_field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.role, self.message, frozenset(self.metadata.items())))


ChatTemplate = list[ChatTurn]

####### configuration #######


@dataclass
class DatasetConfig:
    """Configuration shared by materialized datasets.

    Attributes:
        suffix: Optional suffix appended to the registered dataset key when
            locating the materialized directory.
    """

    suffix: str = field("data/suffix", default="")


class DatasetComponent:
    """Configuration contract shared by raw dataset families."""

    DATASET_KEY: ClassVar[str]
    CONFIG: ClassVar[type[Any] | None] = None
    STYLE: ClassVar[DatasetStyle | None] = None

    @classmethod
    def config(cls) -> list[type[Any]]:
        """Return configuration schemas needed by this dataset.

        Returns:
            The common materialization schema followed by the dataset's owned
            source schema, when it has one.
        """

        return [DatasetConfig] + ([cls.CONFIG] if cls.CONFIG is not None else [])


####### dataset #######


class Dataset(DatasetComponent, ABC, Generic[C]):
    STYLE = DatasetStyle.PADDED

    @abstractmethod
    def __getitem__(self, indx: int) -> C: ...

    @abstractmethod
    def __len__(self) -> int: ...


StringDataset = Dataset[str]
ChatTemplateDataset = Dataset[ChatTemplate]


class PretrainingDataset(StringDataset):
    """
    Pretraining dataset, identical to Dataset[str] but is tokenized differently.
    Specifically, this dataset is tokenized irrespective of item boundaries.
    """

    STYLE = DatasetStyle.PMD


####### contrastive datasets #######


class ContrastiveDataset(DatasetComponent, ABC, Generic[C]):
    STYLE = DatasetStyle.CONTRASTIVE

    @abstractmethod
    def __getitem__(self, indx: int) -> Tuple[C, C]: ...

    @abstractmethod
    def __len__(self) -> int: ...


ContrastiveStringDataset = ContrastiveDataset[str]
ContrastiveChatTemplateDataset = ContrastiveDataset[ChatTemplate]

####### streaming datasets #######


class StreamingDataset(DatasetComponent, ABC, Generic[C]):
    STYLE = DatasetStyle.PADDED

    @abstractmethod
    def __iter__(self) -> Iterator[C]: ...


StreamingStringDataset = StreamingDataset[str]
StreamingChatTemplateDataset = StreamingDataset[ChatTemplate]


class StreamingPretrainingDataset(StreamingStringDataset):
    """
    Pretraining dataset, identical to Dataset[str] but is tokenized differently.
    Specifically, this dataset is tokenized irrespective of item boundaries.
    """

    STYLE = DatasetStyle.PMD
