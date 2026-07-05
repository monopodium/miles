"""``sendrecv_broadcast_uccl``: same relay/fanout design as ``sendrecv_broadcast`` but
the cross-node trainer->relay hop uses uccl-p2p one-sided RDMA write instead of NCCL
``isend``. Only Hop 1 (trainer rank0 -> relay TP0) changes; the coordinator, relay->peer
fanout, pause/continue, and async tail are inherited unchanged from the parent.

See ``docs/UCCL_P2P_WEIGHT_SYNC_DESIGN.md``.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state

from .broadcast_utils import acquire_rollout_engine_lock
from .sendrecv_broadcast import UpdateWeightSendRecvBroadcast
from .uccl_utils import (
    connect_rollout_relay_uccl,
    disconnect_rollout_relay_uccl,
    update_weights_from_distributed_uccl,
)


class UpdateWeightSendRecvBroadcastUccl(UpdateWeightSendRecvBroadcast):
    """Relay-based weight update whose trainer->relay transport is uccl-p2p RDMA write."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict | None,
        is_lora: bool = False,
    ) -> None:
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )
        self._uccl_conn = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if len(engine_gpu_counts) != len(rollout_engines):
            raise ValueError(
                "sendrecv_broadcast_uccl requires one GPU count per rollout engine, "
                f"but got {len(engine_gpu_counts)} counts for {len(rollout_engines)} engines."
            )
        relay_gpu_count = engine_gpu_counts[0]
        if any(gpu_count != relay_gpu_count for gpu_count in engine_gpu_counts):
            raise ValueError(
                "sendrecv_broadcast_uccl requires homogeneous rollout engine GPU counts "
                f"for relay fanout, but got {list(engine_gpu_counts)}."
            )

        self._relay_engine = rollout_engines[0]
        self._peer_engines = list(rollout_engines[1:])
        self._relay_gpu_count = relay_gpu_count
        self._pp_rank = get_parallel_state().pp.rank
        self._connect_coordinator()

        if self._is_source:
            self._group_name = f"miles-sendrecv-broadcast-uccl-train-pp_{self._pp_rank}"
            if self._uccl_conn is not None:
                disconnect_rollout_relay_uccl(self._group_name, self._uccl_conn, self._relay_engine)
                self._uccl_conn = None
            self._uccl_conn = connect_rollout_relay_uccl(
                self._group_name,
                self._relay_engine,
                self.args.update_weight_buffer_size,
            )

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        acquire_rollout_engine_lock(self.rollout_engine_lock)
        try:
            relay_receive_ref = update_weights_from_distributed_uccl(
                self._group_name,
                self._uccl_conn,
                self._relay_engine,
                converted_named_tensors,
            )
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        self._relay_receive_refs.append(relay_receive_ref)
        converted_named_tensors.clear()

        if pbar:
            pbar.update(1)
