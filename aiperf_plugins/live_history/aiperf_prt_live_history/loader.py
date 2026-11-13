# SPDX-License-Identifier: Apache-2.0
"""``weka_trace`` loader that tags each replayed assistant message with its source turn.

The built-in loader replays turn ``k``'s output as an ``assistant`` message in
turn ``k+1``'s delta, filled with corpus text for the trace's hash blocks. That
text never matches what the server generated for turn ``k``, so its KV is always
recomputed. This loader adds one key, ``prt_src = "<session_id>#<k>#<tokens>"``, to every
assistant message that stands for the full output of turn ``k`` of chain
``session_id``. :mod:`aiperf_prt_live_history.endpoint` strips the key and, when
it has the server's real output for that turn, sends that instead.

Tagging, all at dataset build time:

* An assistant segment created by ``advance_turn`` at turn ``k`` is turn
  ``k-1``'s output. It keeps its tag when a later pull-back cuts it
  mid-segment: the endpoint then sends the same-length prefix of the real
  output, which is still a prefix of what the server cached.
* A sub-agent / flat chain's turn 0 can repeat an earlier request's assistant
  segments (the framing loader copies that request's role tiling). Those are
  matched by content within the same trace and get the provider's tag, so the
  child renders the same tokens as the provider did.

Builds on :class:`aiperf_weka_framing.loader.FramingWekaTraceLoader` (same role
framing, same prompts otherwise; the extra key is the only change).
"""

from __future__ import annotations

from typing import Any

import aiperf_weka_framing.loader as framing
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import Conversation

_logger = AIPerfLogger(__name__)

SRC_KEY = "prt_src"
_SEED_TURN = ":turn_"


class LiveHistoryReconstructor(framing.FramingReconstructor):
    """Framing reconstructor that remembers which turn each assistant segment answers."""

    def advance_turn(self, *args: Any, **kwargs: Any) -> None:
        """Advance as the base class, then record the source of a new assistant segment.

        Args:
            *args: Passed through.
            **kwargs: Passed through; ``seed`` must be
                ``f"{session_id}:turn_{k}:partial_tail"`` for a tag to be set.
        """
        before = {id(s) for s in self._segments}
        super().advance_turn(*args, **kwargs)
        seed = kwargs.get("seed", args[5] if len(args) > 5 else "")
        if _SEED_TURN not in seed:
            return
        session_id = seed.rsplit(_SEED_TURN, 1)[0]
        for seg in self._segments:
            if seg.role == "assistant" and id(seg) not in before:
                # _turn_index was incremented to k; the segment carries turn k-1's output.
                seg.prt_src = f"{session_id}#{self._turn_index - 1}"

    def turn_delta(self) -> Any:
        """Return the base delta with ``prt_src = "<sid>#<k>#<tokens>"`` on tagged assistant messages.

        ``<tokens>`` is the message's current token count, so the endpoint can
        size the replacement without tokenizing.
        """
        delta = super().turn_delta()
        msgs = delta.delta_messages
        # Without tool shaping, messages map 1:1 onto the trailing segments.
        segs = self._segments[len(self._segments) - len(msgs) :]
        for msg, seg in zip(msgs, segs, strict=True):
            src = getattr(seg, "prt_src", None)
            if seg.role == "assistant" and src is not None:
                msg[SRC_KEY] = f"{src}#{len(seg.tokens)}"
        return delta


def _trace_of(conv: Conversation) -> str:
    """Return the weka trace id of a conversation (child ids are ``{trace}::...``)."""
    return conv.session_id.split("::", 1)[0]


def _tag_inherited(conversations: list[Conversation]) -> int:
    """Tag untagged assistant messages whose text equals a tagged one in the same trace.

    Args:
        conversations: All conversations of the dataset.

    Returns:
        Number of messages tagged here.
    """
    by_trace: dict[str, dict[str, str]] = {}
    for conv in conversations:
        known = by_trace.setdefault(_trace_of(conv), {})
        for turn in conv.turns:
            for msg in turn.raw_messages or []:
                if SRC_KEY in msg:
                    known.setdefault(msg["content"], msg[SRC_KEY])
    n = 0
    for conv in conversations:
        known = by_trace[_trace_of(conv)]
        for turn in conv.turns:
            for msg in turn.raw_messages or []:
                if msg.get("role") == "assistant" and SRC_KEY not in msg:
                    src = known.get(msg.get("content"))
                    if src is not None:
                        msg[SRC_KEY] = src
                        n += 1
    return n


class LiveHistoryWekaTraceLoader(framing.FramingWekaTraceLoader):
    """Framing ``weka_trace`` loader whose assistant messages carry ``prt_src`` tags."""

    def convert_to_conversations(self, data: dict[str, list[Any]]) -> list[Conversation]:
        """Build conversations as the framing loader, with source tags.

        Args:
            data: Trace id to parsed traces.

        Returns:
            The conversations.
        """
        # The framing loader subclasses the module-global FramingReconstructor
        # at call time; point that name at ours for the duration of the build.
        original = framing.FramingReconstructor
        framing.FramingReconstructor = LiveHistoryReconstructor
        try:
            conversations = super().convert_to_conversations(data)
        finally:
            framing.FramingReconstructor = original
        n = _tag_inherited(conversations)
        _logger.info(f"prt_live_history: tagged {n} inherited assistant messages")
        return conversations

    def _reconstruct_parallel(
        self, *, configured_workers: int, **kwargs: Any
    ) -> list[Conversation]:
        # Forkserver workers never import this plugin, so the reconstructor
        # swap cannot reach them; the serial path is byte-identical upstream.
        return self._reconstruct_serial(dropped_per_trace={}, **kwargs)
