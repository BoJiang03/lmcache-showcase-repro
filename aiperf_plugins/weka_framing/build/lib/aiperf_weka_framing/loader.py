# SPDX-License-Identifier: Apache-2.0
"""``weka_trace`` loader that keeps the provider's role framing on shared prefixes.

Within one weka trace, hash ids are prefix-chained: a chain whose turn 0 repeats
blocks ``[0, L)`` of an earlier request repeated that request's token prefix,
role framing included. The built-in loader builds every turn 0 as
``[system(tool+system)] + [user(rest)]``. When the earlier request had several
messages in ``[0, L)``, the chat template renders different tokens at its first
``<|im_end|>`` and the server prefix cache misses everything after it.

This loader copies the role tiling of blocks ``[0, L)`` from the earlier request
with the longest such prefix (latest on ties) and puts the remainder in the
trailing user message. Chains whose default framing already matches are left
unchanged, so their prompts are byte-identical to the built-in loader's.

Two compromises, both forced by 0.13.0 having no reconstructor hook:

* The built-in loader constructs ``ConversationReconstructor`` by importing it
  from ``weka_synth_buf`` at call time on every path. The subclass is swapped
  into that module attribute only for the duration of
  :meth:`FramingWekaTraceLoader.convert_to_conversations`.
* Parallel workers are forked from a forkserver that never imports this plugin,
  so the swap cannot reach them. When any layout is inherited, reconstruction
  runs on the serial path, which upstream keeps byte-identical to the parallel
  one.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from aiperf.common.models import Conversation
from aiperf.dataset.loader import weka_synth_buf
from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    compute_asst_block_caps,
    longest_common_prefix,
)
from aiperf.dataset.loader.weka_trace import (
    WekaTraceLoader,
    _child_plans_for_active_subagents,
)

Layout = list[tuple[str, int]]

# Every built-in turn-0 call site seeds with f"{session_id}:turn_0:partial_tail".
_TURN0_SEED_SUFFIX = ":turn_0:partial_tail"


def _cut_layout(layout: Layout, blocks: int) -> Layout:
    """Return the first ``blocks`` blocks of a role tiling, without empty segments.

    Args:
        layout: ``[(role, block_count), ...]`` tiling.
        blocks: Number of leading blocks to keep.

    Returns:
        The truncated tiling.
    """
    out: Layout = []
    for role, n in layout:
        take = min(n, blocks)
        if take > 0:
            out.append((role, take))
        blocks -= take
        if blocks == 0:
            break
    return out


class FramingReconstructor(ConversationReconstructor):
    """Reconstructor whose turn 0 can follow an inherited role tiling.

    ``turn0_layouts`` maps a session id to its inherited tiling. It is set on a
    per-call subclass by :class:`FramingWekaTraceLoader`; the base class value
    is empty, which reproduces the built-in behaviour.
    """

    turn0_layouts: dict[str, Layout] = {}

    def init_turn_0(
        self,
        hash_ids: list[int],
        in_tokens: int,
        tool_tokens: int,
        system_tokens: int,
        seed: str,
        is_tool_result: bool = False,
        *,
        inherit_layout: Layout | None = None,
    ) -> None:
        """Initialize turn 0, following an inherited role tiling when one applies.

        Args:
            hash_ids: Turn-0 hash ids.
            in_tokens: Turn-0 input length.
            tool_tokens: Tool-prefix tokens (ignored when a tiling applies).
            system_tokens: System-prefix tokens (ignored when a tiling applies).
            seed: Partial-tail seed; ``f"{session_id}:turn_0:partial_tail"``.
            is_tool_result: Passed through to the base class.
            inherit_layout: Explicit tiling. When None, it is looked up in
                ``turn0_layouts`` by the session id encoded in ``seed``.

        Raises:
            ValueError: As the base class.
        """
        if inherit_layout is None and seed.endswith(_TURN0_SEED_SUFFIX):
            inherit_layout = self.turn0_layouts.get(seed[: -len(_TURN0_SEED_SUFFIX)])
        if not inherit_layout:
            super().init_turn_0(
                hash_ids, in_tokens, tool_tokens, system_tokens, seed, is_tool_result
            )
            return
        # With no tool/system prefix the base class emits one user segment that
        # holds every covered block plus the synthesized tail; re-tile it.
        super().init_turn_0(hash_ids, in_tokens, 0, 0, seed, is_tool_result)
        (whole,) = self._segments
        bs = self.block_size
        segs: list[RoleSegment] = []
        cursor = 0
        for role, n in inherit_layout:
            tokens = whole.tokens[cursor * bs : (cursor + n) * bs]
            segs.append(
                RoleSegment(
                    role=role,
                    block_start=cursor,
                    block_count=n,
                    tokens=tokens,
                    content=self.decode_tokens_to_text(tokens),
                )
            )
            cursor += n
        rest = whole.tokens[cursor * bs :]
        rest_blocks = whole.block_count - cursor
        if rest and segs[-1].role == "user":
            # Continue the inherited trailing user message rather than opening
            # a second adjacent one; its tokens up to the cut are unchanged.
            last = segs[-1]
            last.block_count += rest_blocks
            last.tokens = last.tokens + rest
            last.content = self.decode_tokens_to_text(last.tokens)
        elif rest:
            segs.append(
                RoleSegment(
                    role="user",
                    block_start=cursor,
                    block_count=rest_blocks,
                    tokens=rest,
                    content=self.decode_tokens_to_text(rest),
                )
            )
        self._segments = segs

    def segment_layout(self) -> Layout:
        """Return the current ``[(role, block_count), ...]`` tiling, in order."""
        return [(s.role, s.block_count) for s in self._segments]


def _plan_turn0_layouts(
    trace: Any, plan: Any, child_plans: list[Any], flat_plans: list[Any]
) -> dict[str, Layout]:
    """Return inherited turn-0 tilings for one trace, keyed by session id.

    Chains are processed in turn-0 time order. Each chain's turn 0 is matched
    against every request already replayed (``t`` strictly earlier); the one
    with the longest hash LCP wins, latest on ties. Layouts come from a
    token-free replay of the same reconstructor, so a provider that itself
    inherited is replayed with its inherited framing.

    Args:
        trace: The ``WekaTrace`` (main chain tool/system tokens).
        plan: The trace's ``_ParentPlan``.
        child_plans: All ``_ChildPlan`` entries (filtered to active ones here).
        flat_plans: The trace's ``_FlatChainPlan`` entries.

    Returns:
        ``{session_id: layout}`` for chains whose framing changes.
    """
    bs = plan.block_size
    chains: list[tuple[str | None, list[Any], int, int]] = [
        (None, [r for _, r in plan.normals], trace.tool_tokens, trace.system_tokens)
    ]
    chains += [
        (fp.session_id, [r for _, r in fp.requests], fp.init_tool_tokens,
         fp.init_system_tokens)
        for fp in flat_plans
    ]
    chains += [
        (cp.session_id, cp.requests, cp.init_tool_tokens, cp.init_system_tokens)
        for cp in _child_plans_for_active_subagents(plan, child_plans)
    ]
    chains = sorted((c for c in chains if c[1]), key=lambda c: c[1][0].t)
    layouts: dict[str, Layout] = {}
    seen: list[tuple[float, list[int], Layout]] = []
    for session_id, reqs, tool_tokens, system_tokens in chains:
        first = reqs[0]
        covered = min(len(first.hash_ids), first.input_length // bs)
        best_lcp, best_t, best_layout = 0, 0.0, None
        for t, hash_ids, layout in seen:
            if t >= first.t:
                continue
            lcp = min(
                longest_common_prefix(hash_ids, first.hash_ids),
                covered,
                sum(n for _, n in layout),
            )
            if lcp > 0 and (lcp > best_lcp or (lcp == best_lcp and t >= best_t)):
                best_lcp, best_t, best_layout = lcp, t, layout
        inherit: Layout | None = None
        if session_id is not None and best_layout is not None:
            inherit = _cut_layout(best_layout, best_lcp)
            prefix_blocks = (
                min(math.ceil((tool_tokens + system_tokens) / bs), covered, best_lcp)
                if tool_tokens > 0 or system_tokens > 0
                else 0
            )
            default = [("system", prefix_blocks)] if prefix_blocks else []
            if best_lcp > prefix_blocks:
                default.append(("user", best_lcp - prefix_blocks))
            has_tail = first.input_length > best_lcp * bs
            if inherit == default or (inherit[-1][0] != "user" and not has_tail):
                inherit = None
            if inherit is not None:
                layouts[session_id] = inherit
        recon = FramingReconstructor(
            block_size=bs,
            decode_block_tokens=lambda h: [0] * (len(h) * bs),
            sample_partial_tail_tokens=lambda n, seed: [0] * n,
            decode_tokens_to_text=lambda tokens: "",
        )
        caps = compute_asst_block_caps([(r.hash_ids, r.input_length) for r in reqs], bs)
        for k, req in enumerate(reqs):
            if k == 0:
                recon.init_turn_0(
                    hash_ids=req.hash_ids,
                    in_tokens=req.input_length,
                    tool_tokens=tool_tokens,
                    system_tokens=system_tokens,
                    seed="",
                    inherit_layout=inherit,
                )
            else:
                prev = reqs[k - 1]
                recon.advance_turn(
                    prev_hash_ids=prev.hash_ids,
                    prev_in_tokens=prev.input_length,
                    prev_out_tokens=prev.output_length,
                    curr_hash_ids=req.hash_ids,
                    curr_in_tokens=req.input_length,
                    seed="",
                    max_asst_blocks=caps[k],
                )
            seen.append((req.t, req.hash_ids, recon.segment_layout()))
    return layouts


class FramingWekaTraceLoader(WekaTraceLoader):
    """Built-in ``weka_trace`` loader with provider-framing inheritance on turn 0."""

    def convert_to_conversations(self, data: dict[str, list[Any]]) -> list[Conversation]:
        """Build conversations as the base class, with inherited turn-0 framing.

        Args:
            data: Trace id to parsed traces, as the base class.

        Returns:
            The conversations.
        """
        self._turn0_layouts: dict[str, Layout] = {}
        bound = type(
            "BoundFramingReconstructor",
            (FramingReconstructor,),
            {"turn0_layouts": self._turn0_layouts},
        )
        original = weka_synth_buf.ConversationReconstructor
        weka_synth_buf.ConversationReconstructor = bound
        try:
            return super().convert_to_conversations(data)
        finally:
            weka_synth_buf.ConversationReconstructor = original

    def _build_reconstruction_plans(self, data: dict[str, list[Any]]) -> Any:
        plans = super()._build_reconstruction_plans(data)
        flats_by_trace: dict[str, list[Any]] = defaultdict(list)
        for fp in plans.flat_plans:
            flats_by_trace[fp.parent_trace_id].append(fp)
        layouts: dict[str, Layout] = {}
        for plan in plans.parent_plans:
            layouts.update(
                _plan_turn0_layouts(
                    data[plan.trace_id][0],
                    plan,
                    plans.child_plans,
                    flats_by_trace[plan.trace_id],
                )
            )
        self._turn0_layouts.update(layouts)
        return plans

    def _reconstruct_parallel(
        self, *, configured_workers: int, **kwargs: Any
    ) -> list[Conversation]:
        if not self._turn0_layouts:
            return super()._reconstruct_parallel(
                configured_workers=configured_workers, **kwargs
            )
        return self._reconstruct_serial(dropped_per_trace={}, **kwargs)
