"""The UCCL-P2P `NativeTransport` of `P2PTransport`, with LZip-P2P lossless compression.

CUDA/ROCm only: `--p2p-transport uccl` needs GPU transfer buffers and GPUDirect RDMA, and this module is
imported only when that transport is selected.

With ``UCCL_P2P_COMPRESS_STRATEGY=split_only`` on both the trainer and the rollout engines (UCCL built with
``USE_DIETGPU=1``), every float weight of at least ``UCCL_P2P_MIN_COMPRESS_BYTES`` is compressed on the trainer
GPU with the split-send pipeline, written into the engine's staging arena, and decompressed by UCCL into the
published weight. Compression is lossless, so the engine receives bit-identical weights. Without that variable
UCCL-P2P writes the bytes as they are, which is the uncompressed baseline.
"""

import base64
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import RemoteShard
from miles.platforms import current_platform

_COMPRESS_STRATEGY = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "none").lower()
# the defaults UCCL itself applies (p2p/rdma/compression.cc)
_MIN_COMPRESS_BYTES = int(os.environ.get("UCCL_P2P_MIN_COMPRESS_BYTES", 16 << 20))
_COMPRESS_BUFFER_BYTES = int(os.environ.get("UCCL_P2P_COMPRESS_BUFFER_BYTES", 2 << 30))


@dataclass(frozen=True)
class _UcclPeer:
    conn_id: int
    descriptor_by_name: dict[str, bytes]
    dtype_by_name: dict[str, str]


class UcclTransport:
    """One UCCL-P2P endpoint for this trainer process.

    Each runner of a rollout engine rank publishes a UCCL endpoint and a 64-byte write descriptor per weight;
    `connect_peer` opens an RDMA connection to it. A write is one RDMA WRITE per weight, followed by a fence that
    returns once the engine has decompressed every compressed weight into place.
    """

    name = "uccl"
    # GPU memory: the compressor reads the weights on this GPU and the NIC reads them over GPUDirect RDMA
    transfer_buffer_device = torch.device(current_platform.device_type)

    def __init__(self, write_timeout: float) -> None:
        # not in miles' requirements; torch must be imported before uccl.p2p on ROCm
        from uccl import p2p

        self._p2p = p2p
        self._float_types = {
            "bfloat16": p2p.FloatType.kBFloat16,
            "float16": p2p.FloatType.kFloat16,
            "float32": p2p.FloatType.kFloat32,
            "float8_e4m3fn": p2p.FloatType.kFloat8E4M3FN,
            "float8_e5m2": p2p.FloatType.kFloat8E5M2,
        }
        self._endpoint = p2p.Endpoint(torch.accelerator.current_device_index())
        self._write_timeout = write_timeout
        self._compressing = _COMPRESS_STRATEGY in ("split", "split_only")
        self._peers: dict[str, _UcclPeer] = {}
        # registered transfer buffers: start address -> (nbytes, mr id)
        self._buffer_mrs: dict[int, tuple[int, int]] = {}
        # compression contexts are bound to an exact (address, nbytes, float type); buffer layouts repeat every
        # sync, so each weight slot is prepared once
        self._compress_mr_by_slot: dict[tuple[int, int, str], int] = {}
        self._lock = threading.Lock()
        # UCCL has one compressor per process: its staging buffer is reused by the next compressed write as soon
        # as the previous one completes, so compressed writes of all write threads run one at a time
        self._compress_lock = threading.Lock()
        # a write that did not finish does not prove UCCL stopped reading its source
        self._has_failed_writes = False

    def connect_peer(self, transfer_engine_info: dict) -> str:
        metadata = base64.b64decode(transfer_engine_info["endpoint_metadata"])
        ip, port, gpu = self._p2p.Endpoint.parse_metadata(metadata)
        # always RDMA, also to a runner on this host: compression lives on the RDMA path
        ok, conn_id = self._endpoint.connect(ip, gpu, remote_port=port)
        if not ok:
            raise RuntimeError(f"UCCL could not connect to the rollout runner endpoint at {ip}:{port}")
        peer = f"{ip}:{port}"
        with self._lock:
            self._peers[peer] = _UcclPeer(
                conn_id=conn_id,
                descriptor_by_name={
                    name: base64.b64decode(blob) for name, blob in transfer_engine_info["weight_descriptors"].items()
                },
                dtype_by_name=dict(transfer_engine_info["weight_dtypes"]),
            )
        return peer

    def remove_peers(self) -> None:
        with self._lock:
            peers, self._peers = self._peers, {}
        for peer in peers.values():
            self._endpoint.remove_remote_endpoint(peer.conn_id)

    def register_tensor(self, tensor: torch.Tensor) -> Any:
        ok, mr_id = self._endpoint.reg(tensor.data_ptr(), tensor.nbytes)
        if not ok:
            raise RuntimeError(f"UCCL could not register {tensor.nbytes} bytes at {tensor.data_ptr():#x}")
        with self._lock:
            self._buffer_mrs[tensor.data_ptr()] = (tensor.nbytes, mr_id)
        return tensor.data_ptr()

    def unregister_tensor(self, address: int) -> None:
        if self._has_failed_writes:
            raise RuntimeError(
                f"Cannot release registered memory after a {self.name} write with unknown native completion"
            )
        with self._lock:
            nbytes, mr_id = self._buffer_mrs.pop(address)
            slots = [slot for slot in self._compress_mr_by_slot if address <= slot[0] < address + nbytes]
            compress_mr_ids = [self._compress_mr_by_slot.pop(slot) for slot in slots]
        for compress_mr_id in compress_mr_ids:
            self._endpoint.dereg(compress_mr_id)
        self._endpoint.dereg(mr_id)

    def write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        peer = self._peers[remote_shard.transport_peer]
        succeeded = False
        try:
            for name, tensor in tensors_by_name.items():
                if tensor.nbytes == 0:
                    continue
                self._write_one(peer, name, tensor)
            # a compressed write completes when its bytes reach the engine's staging arena; the weights hold the
            # data only once UCCL decompressed them, which this waits for
            if not self._endpoint.flush_compressed_writes(peer.conn_id, int(self._write_timeout * 1000)):
                raise TimeoutError(f"UCCL decompression on {remote_shard.transport_peer} did not finish in time")
            succeeded = True
        finally:
            # write threads share this flag: only ever set it, so one thread's success never clears another's failure
            if not succeeded:
                self._has_failed_writes = True

    def _write_one(self, peer: _UcclPeer, name: str, tensor: torch.Tensor) -> None:
        descriptor = peer.descriptor_by_name[name]
        compress_mr_id = self._compress_mr_id(tensor, peer.dtype_by_name[name])
        if compress_mr_id is None:
            ok = self._endpoint.write(peer.conn_id, self._buffer_mr_id(tensor), tensor.data_ptr(), tensor.nbytes, descriptor)
        else:
            with self._compress_lock:
                ok = self._endpoint.write(peer.conn_id, compress_mr_id, tensor.data_ptr(), tensor.nbytes, descriptor)
        if not ok:
            raise RuntimeError(f"the UCCL write of {name} ({tensor.nbytes} bytes) failed")

    def _compress_mr_id(self, tensor: torch.Tensor, dtype: str) -> int | None:
        """The compress-only registration of exactly this weight slot, or None to write it uncompressed."""
        if not (
            self._compressing
            and dtype in self._float_types
            and _MIN_COMPRESS_BYTES <= tensor.nbytes <= _COMPRESS_BUFFER_BYTES
            and tensor.nbytes % 2 == 0
        ):
            return None
        slot = (tensor.data_ptr(), tensor.nbytes, dtype)
        with self._lock:
            mr_id = self._compress_mr_by_slot.get(slot)
        if mr_id is None:
            ok, mr_id = self._endpoint.reg_compress_only(tensor.data_ptr(), tensor.nbytes, self._float_types[dtype])
            if not ok:
                raise RuntimeError(f"UCCL could not prepare compression for {tensor.nbytes} bytes of {dtype}")
            with self._lock:
                self._compress_mr_by_slot[slot] = mr_id
        return mr_id

    def _buffer_mr_id(self, tensor: torch.Tensor) -> int:
        address = tensor.data_ptr()
        for start, (nbytes, mr_id) in self._buffer_mrs.items():
            if start <= address and address + tensor.nbytes <= start + nbytes:
                return mr_id
        raise RuntimeError(f"{tensor.nbytes} bytes at {address:#x} are not inside a registered transfer buffer")
