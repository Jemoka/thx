"""
Structures supporting the experiment DAG.
Handles evaluation, branching, etc.
"""

import uuid
import base64
from pydantic import BaseModel, Field

from typing import Self, Optional


class Node(BaseModel):
    """a stable identifier for a particular node"""

    name: str = Field(description="name of the artifact, can.be.keyed")
    nonce: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:6],
        description="unique identifier for this node",
    )
    seq: int = Field(default=0, description="sequence number of this node")
    parent: Optional[str] = Field(default=None, description="the parent node")

    def serialize(self) -> str:
        """serialize the node to a string"""
        return (
            base64.urlsafe_b64encode(f"{self.name}:{self.nonce}:{self.seq}".encode())
            .rstrip(b"=")
            .decode("ascii")
        )

    @classmethod
    def deserialize(cls, s: str) -> Self:
        """deserialize a node from a string"""
        decoded = base64.urlsafe_b64decode(s + "==").decode("ascii")
        name, nonce, seq = decoded.rsplit(":", 2)
        return cls(name=name, nonce=nonce, seq=int(seq))

    def next(self) -> "Node":
        """return the next node in the sequence.

        You should NOT call this method unless you know what you are doing.
        The coherency of this method is important and if you tick accidentally
        it may break the DAG (e.g., either creating a node that's not stored
        or creating extra nodes.)
        """
        return Node(
            name=self.name, nonce=self.nonce, seq=self.seq + 1, parent=self.serialize()
        )

    def update(self, node: "Node") -> None:
        """Update the live clock without replacing references held by consumers."""
        self.name = node.name
        self.nonce = node.nonce
        self.parent = node.parent
        self.seq = node.seq
