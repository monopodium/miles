"""Trainer-side uccl-p2p transport for the ``sendrecv_broadcast_uccl`` mode.

This is the writer half of the one-sided RDMA-write transport that replaces the
cross-node NCCL ``isend`` of ``sendrecv_broadcast`` (see
``docs/UCCL_P2P_WEIGHT_SYNC_DESIGN.md``). The trainer (rank 0) is the uccl *client*
= *writer*; the relay TP0 is the *server* = *receiver*.

Per bucket the trainer:
  1. flattens the bucket into one contiguous uint8 blob (unchanged wire format),
  2. copies it into a persistent, pre-registered staging slot,
  3. RDMA-writes the slot into the relay's pre-advertised ``arena[slot]``, polling to
     completion (this is the only place the trainer blocks),
  4. fires the metadata RPC — which, arriving strictly after the write completed,
     is simultaneously the metadata delivery and the completion signal.
"""

import base64
import json
import logging
import os
import time
from collections import deque

import ray
import torch
from ray.actor import ActorHandle

logger = logging.getLogger(__name__)

# Ring depth and per-slot headroom. Must match what the relay allocates; the trainer
# passes num_slots/slot_bytes to the relay at connect time so both agree.
# v2: 4 slots so the slot-gating ray.get is a no-op in steady state (the relay is
# several buckets ahead), and 1.5x per-slot headroom to hold a single oversized tensor
# (e.g. an 8B embedding ~1.24 GB) without a 2x memory blowup on the relay GPU.
DEFAULT_NUM_SLOTS = 4
# Set MILES_UCCL_PROFILE=1 to log a per-bucket gate/copy/write/rpc timing breakdown.
_PROFILE = os.environ.get("MILES_UCCL_PROFILE", "0") == "1"
# uccl one-sided write hits its fast path (line rate) only when the transfer size is a
# multiple of kMaxSplitNum(16) * page(4096) = 64 KB; otherwise it drops ~5x. Round every
# write up to this. See update_weights_from_distributed_uccl and run/bench_sizesweep.py.
_WRITE_ALIGN = 65536


def _slot_bytes_for(update_weight_buffer_size: int) -> int:
    return update_weight_buffer_size + update_weight_buffer_size // 2  # 1.5x


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _parse_uccl_meta_message(message: str) -> dict:
    for tok in message.split():
        if tok.startswith("UCCLMETA:"):
            payload = json.loads(_b64d(tok[len("UCCLMETA:") :]).decode("ascii"))
            return {
                "meta": _b64d(payload["meta"]),
                "fifo_blobs": [_b64d(fb) for fb in payload["fifo_blobs"]],
                "num_slots": payload["num_slots"],
                "slot_bytes": payload["slot_bytes"],
            }
    raise RuntimeError(f"uccl trainer: no UCCLMETA token in relay init message: {message!r}")


class UcclTrainerConn:
    """Holds the trainer-side uccl endpoint, staging ring, and cached relay fifo blobs."""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.ep = None
        self.conn_id = None
        self.num_slots = 0
        self.slot_bytes = 0
        self.staging: list[torch.Tensor] = []
        self.src_mr_ids: list[int] = []
        self.fifo_blobs: list[bytes] = []  # relay arena advertisements (dest addr+rkey)
        self.bucket_cursor = 0
        self.recent_refs: deque = deque()

    def close(self) -> None:
        try:
            if self.ep is not None:
                for mr_id in self.src_mr_ids:
                    try:
                        self.ep.dereg(mr_id)
                    except Exception:
                        pass
        finally:
            self.ep = None
            self.conn_id = None
            self.staging = []
            self.src_mr_ids = []
            self.fifo_blobs = []
            self.recent_refs.clear()


def connect_rollout_relay_uccl(
    group_name: str,
    relay_engine: ActorHandle,
    update_weight_buffer_size: int,
    num_slots: int = DEFAULT_NUM_SLOTS,
) -> UcclTrainerConn:
    """Stand up the trainer->relay uccl connection. Built once, reused every step.

    Order matters: the relay must create its endpoint + advertise its arena and hand
    back its metadata BEFORE the trainer can ``connect`` (uccl write requires the writer
    to be the connecting side). So we RPC the relay first, then connect.
    """
    from uccl import p2p  # lazy import

    slot_bytes = _slot_bytes_for(update_weight_buffer_size)
    conn = UcclTrainerConn(group_name)
    conn.ep = p2p.Endpoint(torch.cuda.current_device())

    # 1. Ask the relay to build its receiver (endpoint + registered/advertised arena).
    #    master_address/master_port are unused for uccl; pass placeholders.
    response = ray.get(
        relay_engine.init_relay_weights_update_group.remote(
            "",
            0,
            1,
            2,
            group_name,
            backend="nccl",
            transport="uccl",
            num_slots=num_slots,
            slot_bytes=slot_bytes,
        )
    )
    if not isinstance(response, dict) or not response.get("success"):
        raise RuntimeError(f"uccl relay init failed: {response}")
    info = _parse_uccl_meta_message(response["message"])
    conn.num_slots = info["num_slots"]
    conn.slot_bytes = info["slot_bytes"]
    conn.fifo_blobs = info["fifo_blobs"]
    if len(conn.fifo_blobs) != conn.num_slots:
        raise RuntimeError("uccl relay returned mismatched fifo blob count")

    # 2. Register the trainer's persistent staging slots (RDMA source buffers).
    for _ in range(conn.num_slots):
        buf = torch.empty(conn.slot_bytes, dtype=torch.uint8, device=torch.cuda.current_device())
        ok, mr_id = conn.ep.reg(buf.data_ptr(), conn.slot_bytes)
        if not ok:
            raise RuntimeError("uccl trainer: reg() of staging buffer failed")
        conn.staging.append(buf)
        conn.src_mr_ids.append(mr_id)

    # 3. Connect to the relay (drives the relay's accept()).
    ip, port, gpu_bdf = p2p.Endpoint.parse_metadata(info["meta"])
    ok, conn_id = conn.ep.connect(ip, gpu_bdf, remote_port=port)
    if not ok:
        raise RuntimeError(f"uccl trainer: connect to relay {ip}:{port} failed")
    conn.conn_id = conn_id
    return conn


def disconnect_rollout_relay_uccl(
    group_name: str, conn: UcclTrainerConn, relay_engine: ActorHandle
) -> None:
    try:
        ray.get(relay_engine.destroy_relay_weights_update_group.remote(group_name))
    finally:
        conn.close()


def update_weights_from_distributed_uccl(
    group_name: str,
    conn: UcclTrainerConn,
    relay_engine: ActorHandle,
    converted_named_tensors,
) -> ray.ObjectRef:
    """RDMA-write one bucket into the relay arena, then RPC the metadata/completion.

    Returns the Ray ObjectRef of the relay's receive RPC, which resolves only after the
    relay has copied the arena slot out — so it both gates fanout and (K-deep) gates the
    trainer from overwriting a slot the relay has not yet consumed.
    """
    i = conn.bucket_cursor
    slot = i % conn.num_slots
    conn.bucket_cursor += 1
    t0 = time.perf_counter()

    # Ring flow control: before reusing arena[slot], wait for the relay to have copied
    # out the bucket that previously occupied this slot (i - num_slots). With num_slots=4
    # the relay is normally several buckets ahead, so this ray.get is already resolved.
    if len(conn.recent_refs) >= conn.num_slots:
        ray.get(conn.recent_refs.popleft())
    t_gate = time.perf_counter()

    # v2 zero-copy source: write each tensor's bytes DIRECTLY into the pre-registered
    # staging slot, byte-identical to FlattenedTensorBucket's layout (sequential concat of
    # tensor.flatten().view(uint8), no padding). This drops the torch.cat allocation + the
    # separate full-buffer staging copy of v1 (one pass instead of two).
    nbytes = sum(t.numel() * t.element_size() for _, t in converted_named_tensors)
    if nbytes > conn.slot_bytes:
        raise RuntimeError(
            f"uccl trainer: bucket {nbytes} B exceeds slot size {conn.slot_bytes} B; "
            "raise DEFAULT_NUM_SLOTS headroom or lower --update-weight-buffer-size."
        )
    staging = conn.staging[slot]
    offset = 0
    for _name, t in converted_named_tensors:
        n = t.numel() * t.element_size()
        staging[offset : offset + n].copy_(t.flatten().view(torch.uint8))
        offset += n
    # Ensure the copies are done before the NIC reads the staging buffer.
    torch.cuda.current_stream().synchronize()
    t_copy = time.perf_counter()

    # uccl fast-path alignment: the engine splits every message into kMaxSplitNum(=16)
    # chunks and RDMA-writes them in parallel; each chunk is only page(4 KB)-aligned
    # when message_size is a multiple of 16*4096 = 64 KB. Unaligned sizes fall onto a
    # ~5x slower RDMA path (49.5 -> ~10.7 GB/s, verified in run/bench_sizesweep.py).
    # Weight buckets are arbitrary byte sums, so round the transfer up to the next 64 KB
    # (the relay reads only the true `total_bytes`; the padding bytes land in the arena
    # tail and are never read). Staging has 1.5x headroom so the padded write always fits.
    write_nbytes = (nbytes + _WRITE_ALIGN - 1) // _WRITE_ALIGN * _WRITE_ALIGN
    if write_nbytes > conn.slot_bytes:
        write_nbytes = nbytes  # pathological: no headroom for padding, fall back unpadded

    ok, transfer_id = conn.ep.write_async(
        conn.conn_id,
        conn.src_mr_ids[slot],
        staging.data_ptr(),
        write_nbytes,
        conn.fifo_blobs[slot],
    )
    if not ok:
        raise RuntimeError("uccl trainer: write_async failed to enqueue")
    done = False
    while not done:
        ok, done = conn.ep.poll_async(transfer_id)
        if not ok:
            raise RuntimeError("uccl trainer: poll_async reported failure")
    t_write = time.perf_counter()

    # The write has landed in relay arena[slot]; the RPC below is the completion signal.
    ref = relay_engine.update_relay_weights_from_distributed.remote(
        names=[name for name, _ in converted_named_tensors],
        dtypes=[param.dtype for _, param in converted_named_tensors],
        shapes=[param.shape for _, param in converted_named_tensors],
        group_name=group_name,
        load_format="flattened_bucket_uccl",
        slot=slot,
    )
    conn.recent_refs.append(ref)

    if _PROFILE:
        t_rpc = time.perf_counter()
        logger.info(
            "[uccl-bkt %d] %.0fMB gate=%.1f copy=%.1f write+poll=%.1f rpc=%.1f ms (%.1f GB/s wire)",
            i, nbytes / 1e6, 1e3 * (t_gate - t0), 1e3 * (t_copy - t_gate),
            1e3 * (t_write - t_copy), 1e3 * (t_rpc - t_write),
            nbytes / (t_write - t_copy) / 1e9 if t_write > t_copy else 0.0,
        )
    return ref
