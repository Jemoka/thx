"""Append-only object storage for Theseus experiment nodes.

Scalar metadata is written asynchronously into ClickHouse, while blobs are exposed as
directories whose relative locations are recorded in the same metadata stream.
``RecordStore`` adds deliberately simple, process-0-only MessagePack artifacts; it
is not a checkpointing implementation. Stateful jobs should use
``CheckpointedJob`` for distributed state, configuration, and job metadata.
"""

import errno
import fcntl
import os
import re
from base64 import b64encode
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from queue import Empty, Queue
from tempfile import NamedTemporaryFile
from threading import Lock, RLock, Thread
from time import monotonic, sleep, time_ns
from typing import (
    Any,
    Iterator,
    IO,
    Literal,
    Mapping,
    Self,
    Sequence,
    cast,
    overload,
)
from typing_extensions import TypedDict
from uuid import UUID, uuid4
from urllib.parse import unquote, urlsplit

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import OperationalError
import jax
import pyarrow as pa
from flax import serialization

from theseus.base import JobSpec, Node, PyTree
from theseus.base.hardware import Cluster, HardwareResult, local as local_hardware


_VALUE_BATCH_SIZE = 4096
_VALUE_FLUSH_SECONDS = 30.0
_METADATA_TYPES: dict[str, str] = {
    "_x_blob": "String",
    "_x_checkpoint": "Bool",
    "_x_execution": "String",
    "_x_event": "String",
    "_x_finished": "Bool",
    "_x_job": "String",
    "_x_name": "String",
    "_x_nonce": "String",
    "_x_parent": "String",
    "_x_prc_idx": "Int64",
    "_x_record": "Bool",
    "_x_seq": "Int64",
    "_x_tag": "String",
    "_x_timestamp": "Int64",
    "_x_write": "Int64",
}

ValueRow = dict[str, float | int | str | bool | Path | None]


class SerializedQuery(TypedDict):
    """Serializable state produced by a query builder."""

    filters: list[str]
    predicates: list[tuple[str, str | None, str | None]]
    params: dict[str, float | int | str]
    jobs: list[str]
    artifact: bool
    checkpoint: bool
    execution: str | None
    finished: bool | None
    latest: bool
    sorting: tuple[str, bool] | None
    tag: str | None


class ObjectReader:
    """Query ClickHouse metadata and resolve blobs against an object root.

    ``Cluster.metadata`` is a persistent chDB directory/URI or an HTTP(S)
    ClickHouse DSN. Omission selects ``objects/values/v2`` under the cluster root. ``ROOT/.id`` identifies
    the metadata table and must already exist when opening a reader. Embedded
    connections hold an exclusive filesystem lock until close; use a server for
    concurrent processes.
    """

    def __init__(self, cluster: Cluster):
        identity = UUID((cluster.root_dir / ".id").read_text().strip())
        self._table = f"values_{identity.hex}"
        self._object_dir = cluster.objects_dir
        self._metadata = cluster.metadata or str(self._object_dir / "values" / "v2")
        self._connection_lock = RLock()
        self._file_lock: IO[bytes] | None = None
        self._client: Client | None = None
        ObjectReader.start(self)

    @classmethod
    def local(cls, root_dir: str | Path, metadata: str | None = None) -> Self:
        """Read a local object root using the selected metadata connection."""
        return cls(
            Cluster(name="local", root=str(root_dir), work="-", metadata=metadata)
        )

    def start(self) -> None:
        """Open a connection, failing immediately if an embedded path is owned."""
        with self._connection_lock:
            if self._client is not None:
                return
            options: dict[str, Any] = {"dsn": self._metadata}
            parsed = urlsplit(self._metadata)
            if parsed.scheme in {"", "chdb"}:
                if parsed.scheme and (parsed.netloc or parsed.query or parsed.fragment):
                    raise ValueError(
                        "embedded metadata must be a persistent chdb:///path"
                    )
                path = Path(unquote(parsed.path) if parsed.scheme else self._metadata)
                path = path.expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = Path(str(path) + ".lock").open("a+b")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    raise RuntimeError(
                        f"metadata database {path} is already open; "
                        "configure Cluster.metadata with a ClickHouse server "
                        "for concurrent processes"
                    ) from None
                self._file_lock = lock
                options = {"interface": "chdb", "path": str(path)}
            elif parsed.scheme not in {"http", "https"}:
                raise ValueError(
                    "metadata must be a chDB path or an HTTP(S) ClickHouse DSN"
                )
            else:
                options["query_retries"] = 2
            try:
                if self._file_lock is not None:
                    path.mkdir(parents=True, exist_ok=True)
                self._client = clickhouse_connect.get_client(**options)
            except Exception:
                if self._file_lock is not None:
                    self._file_lock.close()
                    self._file_lock = None
                raise

    def close(self) -> None:
        """Close the connection and release embedded database ownership."""
        with self._connection_lock:
            try:
                if self._client is not None:
                    self._client.close()
            finally:
                self._client = None
                if self._file_lock is not None:
                    self._file_lock.close()
                    self._file_lock = None

    def query(self, serialized: SerializedQuery | None = None) -> "QueryBuilder":
        """Return a fresh query builder for this reader."""
        return QueryBuilder(self, serialized)

    def root(self) -> Path:
        """Return the object-store root used to resolve blob paths."""
        return self._object_dir

    def _schema(self) -> dict[str, str]:
        with self._connection_lock:
            if self._client is None:
                raise RuntimeError("metadata connection is closed; call start() first")
            if not self._client.command(f"EXISTS TABLE {self._table}"):
                return {}
            return {
                row["name"]: row["type"].removeprefix("Nullable(").removesuffix(")")
                for row in self._query_arrow(
                    f"DESCRIBE TABLE {self._table}"
                ).to_pylist()
            }

    def _snapshot(self) -> tuple[str, set[str]] | None:
        columns = set(self._schema())
        return (self._table, columns) if columns else None

    def _query_arrow(self, sql: str, params: dict[str, Any] | None = None) -> pa.Table:
        with self._connection_lock:
            if self._client is None:
                raise RuntimeError("metadata connection is closed; call start() first")
            return cast(
                pa.Table,
                self._client.query_arrow(
                    sql,
                    parameters=params,
                    settings={"max_query_size": max(262144, len(sql.encode()) + 1024)},
                ),
            )

    def parents(self, node: Node) -> list[Node]:
        """Return the distinct direct parents recorded for an exact node."""
        snapshot = self._snapshot()
        if snapshot is None:
            return []
        source, _ = snapshot
        name = b64encode(node.name.encode()).decode()
        nonce = b64encode(node.nonce.encode()).decode()
        table = self._query_arrow(
            f"""
            SELECT DISTINCT
                _x_parent AS parent
            FROM {source}
            WHERE _x_name = base64Decode('{name}')
                AND _x_nonce = base64Decode('{nonce}')
                AND _x_seq = {node.seq}
            ORDER BY parent
            """,
        )
        return [
            Node.deserialize(row["parent"])
            for row in table.to_pylist()
            if row["parent"] is not None
        ]


class ObjectStore(ObjectReader):
    """Store blobs and scalar metadata associated with experiment DAG nodes.

    Constructing a store immediately starts its asynchronous ClickHouse writer. Values
    become queryable after a size- or time-based flush, or after :meth:`close` drains
    the writer. Concurrent processes require a ClickHouse server; query conflict
    resolution gives lower process indices precedence. A root's stable ``.id``
    is created atomically on its first store construction.

    Args:
        hardware: Hardware allocation whose current process host determines the
            object-store root.
    """

    #### public methods ####

    def __init__(
        self,
        hardware: HardwareResult,
        execution_id: str | None = None,
        tag: str | None = None,
    ) -> None:
        self.hardware = hardware
        cluster = hardware.hosts[jax.process_index()].cluster
        identity = cluster.root_dir / ".id"
        if not identity.exists():
            # Publish complete contents without replacing a competing creator's ID.
            with NamedTemporaryFile(dir=identity.parent, prefix=".id.") as pending:
                pending.write(uuid4().hex.encode())
                pending.flush()
                os.fsync(pending.fileno())
                try:
                    os.link(pending.name, identity)
                except FileExistsError:
                    pass
                directory = os.open(identity.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        super().__init__(cluster)
        self.execution_id = execution_id
        self.tag = tag
        self.buffer: Queue[ValueRow | None] = Queue()
        self.thread: Thread | None = None
        self.error: Exception | None = None
        self._write_lock = Lock()
        self._write_index = time_ns()
        self.start()

    @classmethod
    def local(cls, root_dir: str | Path, metadata: str | None = None) -> Self:
        """Open a local store using a persistent chDB path or ClickHouse DSN."""
        hardware = local_hardware(str(root_dir), "-")
        hardware.hosts[0].cluster.metadata = metadata
        return cls(hardware)

    def close(self) -> None:
        """Drain queued values and release the connection, including on failure."""
        with self._write_lock:
            if self.thread is None:
                super().close()
                return
            self.buffer.put(None)
            try:
                self.thread.join()
                self._raise_writer_error()
            finally:
                self.thread = None
                super().close()

    #### public writes ####

    # note that for both updating a node with the same keys
    # will just result in the last write winning, so the same operation
    # is idempotent, can be retried, and can be used for C and U

    @contextmanager
    def blob(
        self,
        node: Node,
        metadata: Mapping[str, float | int | str] | None = None,
    ) -> Iterator[Path]:
        """Open the directory used to write a node's blob.

        On normal context exit, process 0 creates ``nodespec.json`` if absent and
        every process records the blob's relative path with the supplied metadata. If the
        context exits with an exception, the partial directory is left in place but
        no metadata row is recorded.

        Args:
            node: Node that owns the blob.
            metadata: Optional scalar metadata to publish with the blob location.

        Yields:
            The directory into which the caller should write blob contents.

        Raises:
            RuntimeError: If the asynchronous value writer has previously failed.
            TypeError: If metadata contains a non-scalar value.
        """
        if metadata is not None:
            self._validate_values(metadata)

        value = self.root() / "blobs" / node.serialize()
        value.mkdir(parents=True, exist_ok=True)
        yield value

        # WARNING WARNING WARNING WARNING
        # THERE'S NO GARANTEES THAT WHEN THE CONTEXT MANAGER
        # RETURNS THAT ANY OF THE FILES HAVE ACTUALLY BEEN WRITTEN
        # EG DUE TO ASNC CHECKPOINTING. YOU MAY NEED TO WAIT
        # OR REFACTOR IF YOU ARE ABOUT TO MOVE THE CHECKPOINT, which
        # we are not doing here but just fyi.

        if jax.process_index() == 0:
            try:
                with open(value / "nodespec.json", "x") as f:
                    f.write(node.model_dump_json())
            except FileExistsError:
                existing = Node.model_validate_json(
                    (value / "nodespec.json").read_text()
                )
                if existing.serialize() != node.serialize():
                    raise ValueError("Blob node specification has a different identity")

        row = dict(metadata) if metadata is not None else {}
        row.update({"_x_blob": os.path.relpath(value, self.root() / "blobs")})

        self.value(node, row)

    def value(self, node: Node, kv: Mapping[str, float | int | str | None]) -> None:
        """Queue a scalar metadata chunk associated with a node.

        The input mapping is copied before internal node, process, parent, and write
        metadata is added. Multiple chunks may target the same node; query
        selection folds their non-conflicting keys together. Keys beginning with
        ``_x_`` are reserved for the store and are not returned by queries.

        Args:
            node: Exact node identity to associate with the values.
            kv: Scalar keys and values to enqueue.

        Raises:
            RuntimeError: If the asynchronous writer has failed.
            TypeError: If ``kv`` contains a non-scalar value.
        """
        self._raise_writer_error()
        self._validate_values(kv)
        with self._write_lock:
            if self.thread is None:
                raise RuntimeError("object store is closed")
            self._write_index = max(self._write_index + 1, time_ns())
            write_index = self._write_index

            row: ValueRow = dict(kv)
            row.update(
                {
                    "_x_name": node.name,
                    "_x_nonce": node.nonce,
                    "_x_prc_idx": jax.process_index(),
                    "_x_seq": node.seq,
                    "_x_parent": node.parent,
                    "_x_execution": self.execution_id,
                    "_x_event": uuid4().hex,
                    "_x_tag": self.tag,
                    "_x_timestamp": write_index,
                    "_x_write": write_index,
                }
            )
            self.buffer.put(row)

    def start(self) -> None:
        """Open the metadata connection and start the batched writer."""
        self._raise_writer_error()
        with self._write_lock:
            if self.thread is not None:
                return
            super().start()
            assert self._client is not None
            columns = ", ".join(
                f"{QueryBuilder._identifier(name)} Nullable({kind})"
                for name, kind in _METADATA_TYPES.items()
            )
            try:
                self._client.command(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ({columns}) "
                    "ENGINE = MergeTree ORDER BY "
                    "(ifNull(_x_name, ''), ifNull(_x_nonce, ''), ifNull(_x_seq, 0))"
                )
            except Exception:
                super().close()
                raise
            self.thread = Thread(target=self._flush_values, daemon=True)
            self.thread.start()

    #### private methods ####

    def _raise_writer_error(self) -> None:
        if self.error is not None:
            raise RuntimeError("value writer failed") from self.error

    def _validate_values(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            if value is not None and not isinstance(value, (Real, str)):
                raise TypeError(
                    f"unsupported value type {type(value)!r} for key {key!r}"
                )

    def _flush_values(self) -> None:
        rows: list[ValueRow] = []
        deadline = monotonic() + _VALUE_FLUSH_SECONDS

        try:
            while True:
                try:
                    row = self.buffer.get(timeout=max(0.0, deadline - monotonic()))
                except Empty:
                    row = None
                    closing = False
                else:
                    closing = row is None
                    if row is not None:
                        rows.append(row)
                        if len(rows) < _VALUE_BATCH_SIZE:
                            continue

                # Keep the same events across retries, including an ambiguous insert.
                for attempt in range(3):
                    try:
                        self._write_values(rows)
                        break
                    except (OSError, OperationalError) as error:
                        transient = (
                            isinstance(error, OSError)
                            and error.errno
                            in {
                                errno.EIO,
                                errno.EINTR,
                                errno.ESTALE,
                                errno.ETIMEDOUT,
                                errno.ECONNRESET,
                                errno.ECONNREFUSED,
                                errno.EPIPE,
                            }
                        ) or (
                            isinstance(error, OperationalError)
                            and any(
                                marker in str(error).lower()
                                for marker in (
                                    "timed out",
                                    "timeout",
                                    "connection reset",
                                    "connection refused",
                                    "connection aborted",
                                    "remote end closed",
                                    "failed to read response data",
                                )
                            )
                        )
                        if not transient or attempt == 2:
                            raise
                        sleep(0.25 * 2**attempt)
                rows.clear()
                deadline = monotonic() + _VALUE_FLUSH_SECONDS
                if closing:
                    return
        except Exception as error:
            self.error = error

    def _write_values(self, rows: list[ValueRow]) -> None:
        if not rows:
            return
        with self._connection_lock:
            schema = self._schema()
            columns: dict[str, str] = {}
            data = []
            for name in sorted({name for row in rows for name in row}):
                values = [row.get(name) for row in rows]
                kinds = {
                    "String"
                    if isinstance(value, str)
                    else "Bool"
                    if isinstance(value, bool)
                    else "Int64"
                    if isinstance(value, Integral)
                    else "Float64"
                    for value in values
                    if value is not None
                }
                if "String" in kinds:
                    inferred = "String"
                elif kinds <= {"Int64", "Float64"}:
                    inferred = "Float64" if "Float64" in kinds else "Int64"
                elif kinds == {"Bool"}:
                    inferred = "Bool"
                else:
                    raise TypeError(f"incompatible scalar types for column {name!r}")
                target = schema.get(name, _METADATA_TYPES.get(name))
                if target is None:
                    if not kinds:
                        raise TypeError(
                            f"cannot infer type of all-null column {name!r}"
                        )
                    target = inferred
                converted: list[float | int | str | bool | None] = []
                for value in values:
                    if value is None:
                        converted.append(None)
                        continue
                    if target == "String":
                        converted.append(
                            str(value).lower()
                            if isinstance(value, bool)
                            else str(value)
                        )
                        continue
                    if isinstance(value, (str, Path)) or isinstance(value, bool) != (
                        target == "Bool"
                    ):
                        raise TypeError(
                            f"column {name!r} has fixed type {target}, got {type(value).__name__}"
                        )
                    if target == "Bool":
                        converted.append(value)
                    elif target == "Int64":
                        number = int(value)
                        if number != value or not -(2**63) <= number < 2**63:
                            raise TypeError(
                                f"unsafe Int64 conversion for column {name!r}: {value}"
                            )
                        converted.append(number)
                    elif target == "Float64":
                        if isinstance(value, Integral) and abs(value) > 2**53:
                            raise TypeError(
                                f"unsafe Float64 conversion for column {name!r}: {value}"
                            )
                        converted.append(float(value))
                    else:
                        raise TypeError(f"unsupported column type {target!r}")
                columns[name] = target
                data.append(converted)
            assert self._client is not None
            additions = [
                f"ADD COLUMN IF NOT EXISTS {QueryBuilder._identifier(name)} "
                f"Nullable({kind})"
                for name, kind in columns.items()
                if name not in schema
            ]
            if additions:
                self._client.command(
                    f"ALTER TABLE {self._table} {', '.join(additions)}",
                    settings={
                        "max_query_size": max(
                            262144,
                            sum(
                                len(name.encode()) * 2 + 128
                                for name in schema.keys() | columns.keys()
                            )
                            + 4096,
                        )
                    },
                )
                # IF NOT EXISTS does not validate a concurrent creator's type.
                schema = self._schema()
                for name, kind in columns.items():
                    if schema[name] != kind:
                        raise TypeError(
                            f"column {name!r} has fixed type {schema[name]}, got {kind}"
                        )
            self._client.insert(
                self._table,
                data,
                column_names=list(columns),
                column_type_names=[f"Nullable({kind})" for kind in columns.values()],
                column_oriented=True,
                settings={
                    "max_query_size": max(
                        262144,
                        sum(len(name.encode()) * 2 + 4 for name in columns) + 4096,
                    ),
                    "async_insert": 0,
                },
            )


class RecordStore(ObjectStore):
    """Write and decode lightweight PyTree artifacts attached to nodes.

    Records are accepted only on one host (JAX process 0) and inefficiently
    serialized as MessagePack there. A bounded worker queue keeps that work off the
    caller's path, but this intentionally simple path is not suitable for
    checkpoints. Stateful jobs should use
    :class:`theseus.job.CheckpointedJob`, which owns distributed Orbax saves along
    with synchronization, randomness, configuration, and job metadata.

    Args:
        hardware: Hardware allocation whose process-0 host writes record payloads.
    """

    #### public methods ####

    def __init__(
        self,
        hardware: HardwareResult,
        execution_id: str | None = None,
        tag: str | None = None,
    ) -> None:
        super().__init__(hardware, execution_id, tag)
        self._record_queue: Queue[Any] | None = None
        self._record_thread: Thread | None = None
        self._record_error: Exception | None = None
        self._start_records()

    def close(self) -> None:
        """Flush records, then flush and close the underlying object store.

        Raises:
            RuntimeError: If the record or scalar writer failed.
            OSError: If record publication fails.
        """
        if self._record_thread is not None and self._record_queue is not None:
            self._record_queue.put(None)
            self._record_thread.join()
            self._record_thread = None
        try:
            self._raise_record_error()
        finally:
            super().close()

    def query(
        self,
        serialized: SerializedQuery | None = None,
    ) -> "_RecordQueryBuilder":
        """Return a fresh query builder that decodes MessagePack records.

        Args:
            serialized: Optional state previously produced for an execution base.

        Returns:
            A record-aware query builder for this store.
        """
        return _RecordQueryBuilder(self, serialized)

    def get_record(self, node: Node | str) -> dict[str, Any]:
        """Return an exact node with any MessagePack records decoded.

        Args:
            node: A node or its serialized identity.

        Returns:
            Folded metadata with decoded MessagePack records added as ``payload``.
            The original blob path remains available as ``blob``.
        """
        values = self.query().node(node).select()
        return values[0] if values else {}

    def artifact(
        self,
        node: Node,
        suffix: str,
        description: Mapping[str, float | int | str],
        payload: PyTree[Any],
    ) -> None:
        """Queue a PyTree artifact owned by an exact node.

        Only one host writes artifacts: calls from JAX processes other than process 0
        are no-ops. Reusing a suffix for the same node replaces that artifact.

        Args:
            node: Exact node that owns the artifact.
            suffix: Unique filename stem within the node's blob directory.
            description: Scalar metadata stored with the artifact marker.
            payload: PyTree serialized as MessagePack on process 0.

        Raises:
            TypeError: If description metadata is not scalar.
            ValueError: If ``suffix`` is not a plain, non-empty filename.
            RuntimeError: If either asynchronous writer has failed.
        """
        if not suffix or Path(suffix).name != suffix:
            raise ValueError("artifact suffix must be a non-empty file name")
        self._validate_values(description)
        if jax.process_index() != 0:
            return
        self.start()
        self._start_records()
        assert self._record_queue is not None
        self._record_queue.put(
            (
                node.model_copy(),
                suffix,
                dict(description) | {"_x_record": True},
                payload,
            )
        )
        self._raise_record_error()

    #### private methods ####

    def _start_records(self) -> None:
        self._raise_record_error()
        if jax.process_index() != 0 or self._record_thread is not None:
            return
        self._record_queue = Queue(maxsize=2)
        self._record_thread = Thread(target=self._write_records, daemon=True)
        self._record_thread.start()

    def _write_records(self) -> None:
        assert self._record_queue is not None
        while (record := self._record_queue.get()) is not None:
            node, suffix, description, payload = record
            try:
                encoded = serialization.to_bytes(jax.device_get(payload))
                with self.blob(node, description) as directory:
                    target = directory / f"{suffix}.msgpack"
                    pending = directory / f".{target.name}.tmp"
                    pending.write_bytes(encoded)
                    pending.replace(target)
            except Exception as error:
                if self._record_error is None:
                    self._record_error = error

    def _raise_record_error(self) -> None:
        if self._record_error is not None:
            raise RuntimeError("record writer failed") from self._record_error

    #### helper methods ####

    @staticmethod
    def _decode(values: ValueRow) -> dict[str, Any]:
        decoded: dict[str, Any] = dict(values)
        blob = decoded.get("blob")
        if isinstance(blob, Path) and (records := sorted(blob.glob("*.msgpack"))):
            decoded["payload"] = [
                serialization.msgpack_restore(record.read_bytes()) for record in records
            ]
        return decoded


class QueryBuilder:
    """Build queries over the nodes recorded by an :class:`ObjectStore`.

    Query methods are cumulative and return the builder, so filters can be chained
    before calling :meth:`all` or :meth:`select`. User values written in separate
    chunks are treated as values of the same node. If a key was written more than
    once, the query uses lower-process and last-write precedence.

    Reserved ``_x_*`` fields cannot be passed to :meth:`where`, :meth:`has`, or
    :meth:`sort`. Use :meth:`name`, :meth:`nonce`, :meth:`seq`, :meth:`job`,
    :meth:`node`, :meth:`checkpoint`, :meth:`artifact`, or :meth:`spec` to query
    supported node metadata.

    Example:
        Start with ``store.query()``, chain filters such as
        ``.spec(group="pretrain").checkpoint()``, then call
        ``.where("eval/score", ">", 0.8).all()`` to return matching nodes.

    Args:
        store: Object store or reader containing the Parquet value parts to query.
        serialized: Optional previously serialized query state.
    """

    #### public methods ####

    def __init__(
        self,
        store: ObjectReader,
        serialized: SerializedQuery | None = None,
    ) -> None:
        self.store = store
        self._filters: list[str]
        self._predicates: list[tuple[str, str | None, str | None]]
        self._params: dict[str, float | int | str]
        self._jobs: list[str]
        self._artifact: bool
        self._checkpoint: bool
        self._execution: str | None
        self._finished: bool | None
        self._latest: bool
        self._sorting: tuple[str, bool] | None
        self._tag: str | None
        self._reset()
        if serialized is not None:
            self._filters = list(serialized["filters"])
            self._predicates = list(serialized["predicates"])
            self._params = dict(serialized["params"])
            self._jobs = list(serialized["jobs"])
            self._artifact = serialized["artifact"]
            self._checkpoint = serialized["checkpoint"]
            self._execution = serialized["execution"]
            self._finished = serialized["finished"]
            self._latest = serialized["latest"]
            self._sorting = serialized["sorting"]
            self._tag = serialized["tag"]

    def node(self, node: Node | str) -> Self:
        """Restrict the query to a serialized node identity.

        Args:
            node: Exact node or a value produced by
                :meth:`theseus.base.Node.serialize`.

        Returns:
            This builder, for continued chaining.
        """
        if isinstance(node, str):
            node = Node.deserialize(node)
        return self.name(node.name).nonce(node.nonce).seq(node.seq)

    def name(self, name: str) -> Self:
        """Restrict the query to an exact node name.

        Args:
            name: Fully qualified node name.

        Returns:
            This builder, for continued chaining.
        """
        self._filters.append(f"_x_name = {self._parameter(name)}")
        return self

    def nonce(self, nonce: str) -> Self:
        """Restrict the query to an exact node nonce.

        Args:
            nonce: Branch nonce shared by nodes in the same lineage.

        Returns:
            This builder, for continued chaining.
        """
        self._filters.append(f"_x_nonce = {self._parameter(nonce)}")
        return self

    def seq(self, seq: int) -> Self:
        """Restrict the query to an exact node sequence number.

        Args:
            seq: Sequence number within a node lineage.

        Returns:
            This builder, for continued chaining.
        """
        self._filters.append(f"_x_seq = {self._parameter(seq)}")
        return self

    def execution(self, execution_id: str) -> Self:
        """Restrict the query to one execution."""
        self._execution = execution_id
        return self

    def tag(self, tag: str) -> Self:
        """Restrict the query to an exact execution tag."""
        self._tag = tag
        return self

    def finished(self, is_finished: bool = True) -> Self:
        """Restrict the query by completion of its execution/tag scope."""
        self._finished = is_finished
        return self

    def latest(self) -> Self:
        """Return only the most recently written matching node."""
        self._latest = True
        return self

    def job(self, job: str) -> Self:
        """Restrict the query to nodes checkpointed by a registered job.

        Args:
            job: Exact registry key recorded in ``_x_job`` checkpoint metadata.

        Returns:
            This builder, for continued chaining.
        """
        self._jobs.append(self._parameter(job))
        return self

    def checkpoint(self) -> Self:
        """Restrict the query to nodes recorded as checkpoints.

        Returns:
            This builder, for continued chaining.
        """
        self._checkpoint = True
        return self

    def artifact(self) -> Self:
        """Restrict the query to nodes containing a recorded PyTree artifact.

        Artifact and checkpoint status are independent, so a node may match both
        :meth:`artifact` and :meth:`checkpoint`.

        Returns:
            This builder, for continued chaining.
        """
        self._artifact = True
        return self

    def spec(
        self,
        project: str | None = None,
        group: str | None = None,
        run: str | None = None,
    ) -> Self:
        """Restrict the query by job-spec name components.

        The node-name convention is delegated to
        :meth:`theseus.job.BasicJob.node_name`. A ``None`` component is left
        unrestricted, so any subset of project, group, and run may be supplied.

        Args:
            project: Project component to match, or ``None`` for any project.
            group: Group component to match, or ``None`` for any group.
            run: Run-name component to match, or ``None`` for any run.

        Returns:
            This builder, for continued chaining.
        """
        from theseus.job import BasicJob

        wildcard = "\0"
        name = BasicJob.node_name(
            JobSpec(
                project=project if project is not None else wildcard,
                group=group if group is not None else wildcard,
                name=run if run is not None else wildcard,
            )
        )
        pattern = re.escape(name).replace(re.escape(wildcard), ".*")
        self._filters.append(f"match(_x_name, {self._parameter(f'^{pattern}$')})")
        return self

    def where(self, key: str, operator: str, value: float | int | str) -> Self:
        """Restrict the query using a comparison against a user value.

        Args:
            key: User metadata key to compare.
            operator: One of ``=``, ``!=``, ``<``, ``<=``, ``>``, or ``>=``.
            value: Scalar value to compare against the folded value of ``key``.

        Returns:
            This builder, for continued chaining.

        Raises:
            ValueError: If ``key`` is reserved or ``operator`` is unsupported.
            TypeError: If ``value`` is not a supported scalar query value.
        """
        if key.startswith("_x_"):
            raise ValueError("reserved fields require a dedicated query method")
        if operator not in {"=", "!=", "<", "<=", ">", ">="}:
            raise ValueError(f"unsupported query operator {operator!r}")
        self._predicates.append((key, operator, self._parameter(value)))
        return self

    def has(self, key: str) -> Self:
        """Restrict the query to nodes containing a user metadata key.

        Args:
            key: User metadata key whose folded value must be non-null.

        Returns:
            This builder, for continued chaining.

        Raises:
            ValueError: If ``key`` is a reserved ``_x_*`` field.
        """
        if key.startswith("_x_"):
            raise ValueError("reserved fields require a dedicated query method")
        self._predicates.append((key, None, None))
        return self

    def sort(self, key: str, ascending: bool = True) -> Self:
        """Order matching nodes by a folded user metadata value.

        Missing values are placed last. Name, nonce, and sequence are used as
        deterministic tie-breakers. Calling this method again replaces the prior
        sort key.

        Args:
            key: User metadata key whose folded value determines the ordering.
            ascending: Whether to place smaller values first.

        Returns:
            This builder, for continued chaining.
        """
        self._sorting = (key, ascending)
        return self

    def all(self) -> list[Node]:
        """Return all matching node identities and clear the query.

        Returns:
            Matching nodes in the requested order. Returns an empty list when the
            store has no finalized parts or a requested key has never been
            recorded.
        """
        try:
            table = self._query(read=False)
        finally:
            self._reset()
        if table is None:
            return []
        return [
            Node(name=row["_x_name"], nonce=row["_x_nonce"], seq=row["_x_seq"])
            for row in table.to_pylist()
        ]

    @overload
    def select(
        self,
        return_nodes: Literal[False] = False,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[ValueRow]: ...

    @overload
    def select(
        self,
        return_nodes: Literal[True],
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[tuple[Node, ValueRow]]: ...

    def select(
        self,
        return_nodes: bool = False,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[ValueRow] | list[tuple[Node, ValueRow]]:
        """Return matching folded values and clear the accumulated query.

        Values use lower-process and last-write precedence. Reserved storage
        metadata is omitted unless ``raw`` is true, while a recorded blob is
        exposed as ``blob`` and resolved against the current object-store root.

        Args:
            return_nodes: Pair every value row with its exact node identity.
            raw: Include selected reserved ``_x_*`` storage metadata.
            keys: Optional columns to project instead of reading every value.

        Returns:
            Matching values in the requested order, optionally paired with their
            nodes. Stores without matching finalized rows return an empty list.
        """
        if not raw and keys is not None and any(key.startswith("_x_") for key in keys):
            raise ValueError("reserved fields require raw=True")
        try:
            table = self._query(read=True, raw=raw, keys=keys)
        finally:
            self._reset()
        if table is None:
            return []

        result: list[tuple[Node, ValueRow]] = []
        for row in table.to_pylist():
            node = Node(
                name=row["_x_name"],
                nonce=row["_x_nonce"],
                seq=row["_x_seq"],
            )
            blob = row.get("_x_blob")
            values = {
                key: value
                for key, value in row.items()
                if value is not None and (raw or not key.startswith("_x_"))
            }
            if blob is not None:
                values["blob"] = self.store.root() / "blobs" / blob
            result.append((node, values))
        if return_nodes:
            return result
        return [values for _, values in result]

    #### private methods ####

    def _reset(self) -> None:
        self._filters = []
        self._predicates = []
        self._params = {}
        self._jobs = []
        self._artifact = False
        self._checkpoint = False
        self._execution = None
        self._finished = None
        self._latest = False
        self._sorting = None
        self._tag = None

    def _query(
        self,
        read: bool,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> pa.Table | None:
        snapshot = self.store._snapshot()
        if snapshot is None:
            return None
        source, columns = snapshot
        requested = set(keys) if keys is not None else None
        physical = {"_x_blob" if key == "blob" else key for key in requested or ()}
        if physical - columns:
            return None
        if any(key not in columns for key, _, _ in self._predicates):
            return None
        if self._jobs and "_x_job" not in columns:
            return None
        if self._artifact and "_x_record" not in columns:
            return None
        if self._checkpoint and "_x_checkpoint" not in columns:
            return None
        if self._execution is not None and "_x_execution" not in columns:
            return None
        if self._tag is not None and "_x_tag" not in columns:
            return None

        filters = list(self._filters)
        if self._execution is not None:
            filters.append(f"_x_execution = {self._parameter(self._execution)}")
        if self._tag is not None:
            filters.append(f"_x_tag = {self._parameter(self._tag)}")
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

        if self._finished is not None:
            scope_finished = False
            if "_x_finished" in columns:
                scope_finished = bool(
                    self.store._query_arrow(
                        f"""
                        SELECT countIf(_x_finished = true) > 0 AS finished
                        FROM {source}
                        {where_sql}
                        """,
                        params=self._params,
                    )
                    .column("finished")[0]
                    .as_py()
                )
            if scope_finished != self._finished:
                return None

        priority = "tuple(-__x_prc_idx, __x_write, __x_event)"
        having: list[str] = []
        for key, operator, parameter in self._predicates:
            identifier = self._identifier(key)
            exists = f"countIf({identifier} IS NOT NULL) > 0"
            if operator is None:
                having.append(exists)
            else:
                having.append(
                    f"({exists} AND argMaxIf({identifier}, {priority}, "
                    f"{identifier} IS NOT NULL) {operator} {parameter})"
                )
        if self._checkpoint:
            having.append("countIf(_x_checkpoint = true) > 0")
        if self._artifact:
            having.append("countIf(_x_record = true) > 0")
        for job in self._jobs:
            having.append(f"countIf(_x_job = {job}) > 0")

        having_sql = f"HAVING {' AND '.join(having)}" if having else ""
        order_sql = "_x_name, _x_nonce, _x_seq"
        limit_sql = ""
        if self._latest:
            latest_write = "max(__x_write)"
            order_sql = f"{latest_write} DESC, _x_seq DESC, _x_name, _x_nonce"
            limit_sql = "LIMIT 1"
        elif self._sorting is not None and self._sorting[0] == "_x_lineage_write":
            lineage_write = "max(max(__x_write))"
            order_sql = (
                f"{lineage_write} OVER (PARTITION BY _x_name, _x_nonce), "
                "_x_seq, _x_name, _x_nonce"
            )
        elif self._sorting is not None and self._sorting[0] in columns:
            identifier = self._identifier(self._sorting[0])
            direction = "ASC" if self._sorting[1] else "DESC"
            order_sql = (
                f"countIf({identifier} IS NOT NULL) = 0, "
                f"argMaxIf({identifier}, {priority}, {identifier} IS NOT NULL) "
                f"{direction}, _x_name, _x_nonce, _x_seq"
            )
        selected = ["_x_name", "_x_nonce", "_x_seq"]
        result_columns = list(selected)
        if read:
            user_columns = {
                key
                for key in columns
                if not key.startswith("_x_") and (requested is None or key in requested)
            }
            raw_columns = {
                key
                for key in columns
                if key.startswith("_x_")
                and key not in {"_x_name", "_x_nonce", "_x_seq", "_x_blob"}
                and raw
                and (requested is None or key in requested)
            }
            folded_columns = sorted(user_columns | raw_columns)
            if "_x_blob" in columns and (
                requested is None or {"blob", "_x_blob"} & requested
            ):
                folded_columns.append("_x_blob")
            if folded_columns:
                selected.append(
                    f"COLUMNS({', '.join(map(self._identifier, folded_columns))}) "
                    f"APPLY(value -> argMaxIf(value, {priority}, value IS NOT NULL))"
                )
                result_columns.extend(folded_columns)
        sql = f"""
            SELECT {", ".join(selected)}
            FROM (
                SELECT *, _x_prc_idx AS __x_prc_idx,
                    _x_write AS __x_write, _x_event AS __x_event
                FROM {source}
            )
            {where_sql}
            GROUP BY _x_name, _x_nonce, _x_seq
            {having_sql}
            ORDER BY {order_sql}
            {limit_sql}
            SETTINGS prefer_column_name_to_alias = 1
            """
        table = self.store._query_arrow(sql, params=self._params)
        # APPLY names outputs after their expressions; restore the public keys.
        return table.rename_columns(result_columns)

    #### helper methods ####

    @staticmethod
    def _identifier(key: str) -> str:
        escaped = key.replace("\\", "\\\\").replace("`", "\\`")
        return f"`{escaped}`"

    def _parameter(self, value: float | int | str) -> str:
        name = f"p{len(self._params)}"
        if isinstance(value, str):
            value_type = "String"
        elif isinstance(value, bool):
            value_type = "Bool"
        elif isinstance(value, int):
            value_type = "Int64"
        elif isinstance(value, Real):
            value_type = "Float64"
            value = float(value)
        else:
            raise TypeError(f"unsupported query value type {type(value)!r}")
        self._params[name] = value
        return f"{{{name}:{value_type}}}"


class _SerializedQueryBuilder(QueryBuilder):
    """Build query state without binding it to an object store."""

    def __init__(self) -> None:
        super().__init__(cast(ObjectReader, None))

    def serialize(self) -> SerializedQuery:
        """Return a detached representation of the accumulated query."""
        return {
            "filters": list(self._filters),
            "predicates": list(self._predicates),
            "params": dict(self._params),
            "jobs": list(self._jobs),
            "artifact": self._artifact,
            "checkpoint": self._checkpoint,
            "execution": self._execution,
            "finished": self._finished,
            "latest": self._latest,
            "sorting": self._sorting,
            "tag": self._tag,
        }

    def _query(
        self,
        read: bool,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> pa.Table | None:
        raise RuntimeError("serialized base queries cannot be executed directly")


class _RecordQueryBuilder(QueryBuilder):
    """Query builder that decodes records selected from a RecordStore."""

    @overload
    def select(
        self,
        return_nodes: Literal[False] = False,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def select(
        self,
        return_nodes: Literal[True],
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[tuple[Node, dict[str, Any]]]: ...

    def select(
        self,
        return_nodes: bool = False,
        raw: bool = False,
        keys: Sequence[str] | None = None,
    ) -> list[dict[str, Any]] | list[tuple[Node, dict[str, Any]]]:
        selected = [
            (node, RecordStore._decode(values))
            for node, values in super().select(return_nodes=True, raw=raw, keys=keys)
        ]
        if return_nodes:
            return selected
        return [values for _, values in selected]
