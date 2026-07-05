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
from collections import deque

import ray
import torch
from ray.actor import ActorHandle

from miles.backends.megatron_utils.sglang import FlattenedTensorBucket

# Ring depth and per-slot headroom. Must match what the relay allocates; the trainer
# passes num_slots/slot_bytes to the relay at connect time so both agree.
DEFAULT_NUM_SLOTS = 2
DEFAULT_SLOT_SIZE_MULT = 2


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
    slot_size_mult: int = DEFAULT_SLOT_SIZE_MULT,
) -> UcclTrainerConn:
    """Stand up the trainer->relay uccl connection. Built once, reused every step.

    Order matters: the relay must create its endpoint + advertise its arena and hand
    back its metadata BEFORE the trainer can ``connect`` (uccl write requires the writer
    to be the connecting side). So we RPC the relay first, then connect.
    """
    from uccl import p2p  # lazy import

    slot_bytes = slot_size_mult * update_weight_buffer_size
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

    # Ring flow control: before reusing arena[slot], wait for the relay to have copied
    # out the bucket that previously occupied this slot (i - num_slots).
    if len(conn.recent_refs) >= conn.num_slots:
        ray.get(conn.recent_refs.popleft())

    flattened_tensor = FlattenedTensorBucket(
        named_tensors=list(converted_named_tensors)
    ).get_flattened_tensor()
    nbytes = flattened_tensor.numel() * flattened_tensor.element_size()
    if nbytes > conn.slot_bytes:
        raise RuntimeError(
            f"uccl trainer: bucket {nbytes} B exceeds slot size {conn.slot_bytes} B; "
            "raise DEFAULT_SLOT_SIZE_MULT or lower --update-weight-buffer-size."
        )

    staging = conn.staging[slot]
    staging_view = staging[:nbytes]
    staging_view.copy_(flattened_tensor.view(torch.uint8).reshape(-1))
    # Ensure the D2D copy is done before the NIC reads the staging buffer.
    torch.cuda.current_stream().synchronize()

    ok, transfer_id = conn.ep.write_async(
        conn.conn_id,
        conn.src_mr_ids[slot],
        staging.data_ptr(),
        nbytes,
        conn.fifo_blobs[slot],
    )
    if not ok:
        raise RuntimeError("uccl trainer: write_async failed to enqueue")
    done = False
    while not done:
        ok, done = conn.ep.poll_async(transfer_id)
        if not ok:
            raise RuntimeError("uccl trainer: poll_async reported failure")

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
    return ref
