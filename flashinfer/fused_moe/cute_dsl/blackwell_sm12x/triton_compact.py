"""Triton kernels for compacting MoE routing IDs.

Remaps global expert IDs to dense local indices (0, 1, 2, ...) for the
micro and static MoE kernels, which expect pre-compacted routing.

Two implementations are provided:

* compact_topk_ids          — micro kernel only (no global_to_local_expert).
  Legacy O(N²) single-block kernel kept for the micro path where total_pairs
  is small (≤40 by the routing cutover threshold).

* compact_topk_ids_with_g2l — full O(N+E) three-phase implementation used by
  the static kernel path.  Produces the same outputs as compact_topk_ids plus
  global_to_local_expert, replacing the CAS+spin logic previously embedded
  inside the static kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Legacy O(N²) kernel — micro path only (total_pairs is small, ≤ 40)
# ---------------------------------------------------------------------------

@triton.jit
def _compact_topk_ids_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    active_expert_count_ptr,
    total_pairs,
    BLOCK: tl.constexpr,
):
    pair_slots = tl.arange(0, BLOCK)
    valid = pair_slots < total_pairs
    ids = tl.load(topk_ids_ptr + pair_slots, mask=valid, other=-1).to(tl.int32)

    row_slots = pair_slots[:, None]
    col_slots = pair_slots[None, :]
    row_valid = valid[:, None]
    col_valid = valid[None, :]

    same_id = ids[:, None] == ids[None, :]
    prior_same = row_valid & col_valid & same_id & (col_slots < row_slots)

    first_flags = valid & (tl.sum(prior_same.to(tl.int32), axis=1) == 0)
    first_prefix = tl.cumsum(first_flags.to(tl.int32), axis=0)

    prior_slots = tl.where(prior_same, col_slots, BLOCK)
    first_match = tl.min(prior_slots, axis=1)
    first_slot = tl.where(first_match < BLOCK, first_match, pair_slots)
    first_slot_mask = col_slots == first_slot[:, None]
    compact_id = tl.sum(tl.where(first_slot_mask, first_prefix[None, :], 0), axis=1) - 1

    tl.store(compact_topk_ids_ptr + pair_slots, compact_id, mask=valid)
    tl.store(weight_expert_ids_ptr + compact_id, ids, mask=valid & first_flags)

    active_expert_count = tl.sum(first_flags.to(tl.int32), axis=0)
    tl.store(active_expert_count_ptr, active_expert_count)


def compact_topk_ids(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    active_expert_count: torch.Tensor,
) -> None:
    """Remap global expert IDs to dense contiguous local indices.

    Args:
        topk_ids: [total_pairs] int32 — flattened global expert IDs.
        compact_topk_ids: [total_pairs] int32 — output: dense local indices.
        weight_expert_ids: [>=total_pairs] int32 — output: local->global map.
        active_expert_count: [1] int32 — output: number of unique experts.
    """
    total_pairs = topk_ids.numel()
    if total_pairs == 0:
        active_expert_count.zero_()
        return
    if compact_topk_ids.numel() < total_pairs:
        raise ValueError("compact_topk_ids must have at least total_pairs elements")
    if active_expert_count.numel() != 1:
        raise ValueError("active_expert_count must have shape [1]")

    block = triton.next_power_of_2(total_pairs)
    num_warps = 1 if block <= 16 else 2
    _compact_topk_ids_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        active_expert_count,
        total_pairs,
        BLOCK=block,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# O(N+E) single-warp implementation — static kernel path
#
# For the static kernel (total_pairs ≤ 640, num_global_experts ≤ 256) a
# single warp of 32 threads is enough. The key insight: we reuse the
# global_to_local_expert output buffer as a zeroed histogram scratch
# during Phase B — it is written as a valid g2l map in Phase C anyway,
# so we can safely read back the counts from it before overwriting.
#
# Phase A: zero global_to_local_expert (reused as hit_count scratch).
# Phase B: each thread covers its strided pairs; tl.atomic_add into
#           global_to_local_expert[expert_id] in global memory (single-CTA
#           so no inter-CTA coherence cost — L1 hits throughout).
# Phase C: one-pass prefix scan over the expert table: assigns dense local
#           ids, writes weight_expert_ids, overwrites global_to_local_expert
#           with the final -1/local_id values, stores active_expert_count.
# Phase D: each thread covers its strided pairs; looks up the final
#           global_to_local_expert values to write compact_topk_ids.
#
# BLOCK_EXPERTS and BLOCK_PAIRS are set to the next-power-of-2 of the
# actual sizes so all tl.load/store are aligned and mask-free in the
# common case.
# ---------------------------------------------------------------------------

@triton.jit
def _compact_topk_ids_with_g2l_kernel(
    topk_ids_ptr,
    compact_topk_ids_ptr,
    weight_expert_ids_ptr,
    global_to_local_expert_ptr,
    active_expert_count_ptr,
    total_pairs,
    num_global_experts,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Phase A: zero global_to_local_expert (used as hit_count scratch)
    # ------------------------------------------------------------------
    e = tl.arange(0, BLOCK_EXPERTS)
    valid_e = e < num_global_experts
    tl.store(global_to_local_expert_ptr + e, tl.zeros([BLOCK_EXPERTS], tl.int32), mask=valid_e)

    tl.debug_barrier()

    # ------------------------------------------------------------------
    # Phase B: histogram into global_to_local_expert scratch
    # ------------------------------------------------------------------
    i = tl.arange(0, BLOCK_PAIRS)
    valid_p = i < total_pairs
    expert_ids = tl.load(topk_ids_ptr + i, mask=valid_p, other=0).to(tl.int32)
    tl.atomic_add(global_to_local_expert_ptr + expert_ids, 1, mask=valid_p)

    tl.debug_barrier()

    # ------------------------------------------------------------------
    # Phase C: prefix scan → overwrite g2l with final values
    # ------------------------------------------------------------------
    counts = tl.load(global_to_local_expert_ptr + e, mask=valid_e, other=0).to(tl.int32)
    first_flag = valid_e & (counts > 0)
    local_ids = tl.cumsum(first_flag.to(tl.int32), axis=0) - 1

    g2l = tl.where(first_flag, local_ids, -1)
    tl.store(global_to_local_expert_ptr + e, g2l, mask=valid_e)
    tl.store(weight_expert_ids_ptr + local_ids, e, mask=valid_e & first_flag)
    tl.store(active_expert_count_ptr, tl.sum(first_flag.to(tl.int32)))

    tl.debug_barrier()

    # ------------------------------------------------------------------
    # Phase D: scatter compact ids
    # ------------------------------------------------------------------
    gids = tl.load(topk_ids_ptr + i, mask=valid_p, other=0).to(tl.int32)
    lids = tl.load(global_to_local_expert_ptr + gids, mask=valid_p, other=0).to(tl.int32)
    tl.store(compact_topk_ids_ptr + i, lids, mask=valid_p)


def compact_topk_ids_with_g2l(
    topk_ids: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    weight_expert_ids: torch.Tensor,
    global_to_local_expert: torch.Tensor,
    active_expert_count: torch.Tensor,
) -> None:
    """O(N+E) compact — single-warp, no scratch buffer.

    Remaps global expert IDs to dense local indices and fills the
    global_to_local_expert reverse map in a single kernel launch.
    Reuses global_to_local_expert as a histogram scratch buffer during
    the kernel, overwriting it with the final map before returning.

    Args:
        topk_ids: [total_pairs] int32 — flattened global expert IDs.
        compact_topk_ids: [total_pairs] int32 — output: dense local indices.
        weight_expert_ids: [>=num_active] int32 — output: local->global map.
        global_to_local_expert: [num_global_experts] int32 — output:
            global->local map (-1 for inactive experts).
        active_expert_count: [1] int32 — output: number of unique experts.
    """
    total_pairs = topk_ids.numel()
    num_global_experts = global_to_local_expert.numel()

    if total_pairs == 0:
        active_expert_count.zero_()
        global_to_local_expert.fill_(-1)
        return

    block_pairs = triton.next_power_of_2(total_pairs)
    block_experts = triton.next_power_of_2(num_global_experts)
    # Use enough warps so that BLOCK_PAIRS / (num_warps * 32) ≤ 1, i.e. each
    # element in the pairs vector maps to a distinct thread. Capped at 4 warps
    # (128 threads) since total_pairs ≤ 640 < 128*5.
    num_warps = min(4, max(1, block_pairs // 32))
    _compact_topk_ids_with_g2l_kernel[(1,)](
        topk_ids,
        compact_topk_ids,
        weight_expert_ids,
        global_to_local_expert,
        active_expert_count,
        total_pairs,
        num_global_experts,
        BLOCK_PAIRS=block_pairs,
        BLOCK_EXPERTS=block_experts,
        num_warps=num_warps,
    )
