"""
Hardware information and topology representation for distributed JAX setups.
Defines a Topology class that encapsulates device and process information,
as well as JAX Mesh configuration.
"""

import jax
import numpy as np
from jax.sharding import Mesh

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict

from theseus.base.axis import Axis, ShardingPolicy
from theseus.base.chip import Chip


class Topology(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    shard: ShardingPolicy = Field(default_factory=ShardingPolicy)

    chip: Annotated[Chip, Field(description="what device is used")]
    device_count: Annotated[int, Field(description="number of devices across cluster")]
    local_device_count: Annotated[int, Field(description="number of devices locally")]
    process_count: Annotated[
        int, Field(description="number of processes running across cluster")
    ]
    is_main: Annotated[
        bool, Field(description="whether this process is the main process")
    ]

    mesh: Annotated[
        Mesh,
        Field(description="JAX Mesh representing the device topology", exclude=True),
    ]

    replicas: Annotated[
        int, Field(description="number of SPMD replicas across cluster")
    ]
    local_replicas: Annotated[
        int, Field(description="number of SPMD replicas per host")
    ]

    @classmethod
    def new(cls, chip: Chip, shard: ShardingPolicy | None = None) -> "Topology":
        """Create a Topology instance based on the current JAX device configuration.

        Args:
            chip: The chip type being used.
            shard: Parallelism policy. The batch axis uses all remaining devices.

        """
        devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        local = jax.local_device_count()

        shard = shard or ShardingPolicy()
        if local % shard.tp != 0:
            raise ValueError(f"tp={shard.tp} must divide local_device_count={local}")

        devices = np.array(devs).reshape(-1, local)
        devices = devices.reshape(-1, shard.tp)

        mesh = Mesh(devices, (Axis.BATCH, Axis.SHARD))

        replicas = jax.device_count() // shard.tp
        local_replicas = local // shard.tp

        assert replicas == mesh.shape[Axis.BATCH]
        assert local_replicas * jax.process_count() == replicas

        return cls(
            shard=shard,
            chip=chip,
            device_count=mesh.size,
            local_device_count=local,
            process_count=jax.process_count(),
            is_main=jax.process_index() == 0,
            mesh=mesh,
            replicas=replicas,
            local_replicas=local_replicas,
        )
