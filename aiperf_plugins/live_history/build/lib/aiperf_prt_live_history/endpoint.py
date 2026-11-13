# SPDX-License-Identifier: Apache-2.0
"""Chat endpoint that replays the server's own output as assistant history.

Pairs with :mod:`aiperf_prt_live_history.loader`, which tags every assistant
message that stands for the full output of a turn with ``prt_src``.

Per request (one endpoint instance per AIPerf worker process):

* ``format_payload`` strips ``prt_src`` from every message. When this worker
  holds the server's output for that source turn of the same session tree, the
  message becomes ``{"role": "assistant", "reasoning": "", "content":
  "<think>\\n" + decode(output_ids)}``, cut to the trace message's token
  length when the output is longer. It also sets ``return_token_ids``.
* ``process_responses`` collects the streamed ``token_ids`` and, for a complete
  PROFILING response, stores them under
  ``(root_correlation_id, conversation_id, turn_index)``.

Why this rendering: the Step template renders an assistant message that comes
before the last plain user message as ``<|im_start|>assistant\\n`` + content,
dropping reasoning. The server generated after the prompt suffix
``<|im_start|>assistant\\n<think>\\n``, so the cached sequence continues with
``<think>\\n`` + output ids. Sending ``reasoning=""`` (vLLM maps it to
``reasoning_content``) stops the template from splitting the content at
``</think>``, so ``<think>\\n`` + the decoded output (special tokens kept) is
rendered verbatim. Exactness then depends only on BPE re-tokenization of the
decoded output, which is checked per response and logged.

Outputs are decoded from token ids rather than reassembled from
``reasoning_content``/``content``, because the reasoning parser strips
``</think>`` and surrounding whitespace and detokenization skips special tokens.

Per-request records go to ``$PRT_LIVE_HISTORY_LOG_DIR/<pid>.jsonl`` and outputs
to ``$PRT_LIVE_HISTORY_LOG_DIR/store/`` (shared by all worker processes); without
the variable, outputs stay per worker process.
"""

from __future__ import annotations

import hashlib
import os
from array import array
from collections import OrderedDict
from typing import Any

import orjson
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ParsedResponse, RequestInfo, RequestRecord, Turn
from aiperf.endpoints.openai_chat import ChatEndpoint

from aiperf_prt_live_history.loader import SRC_KEY

_DEFAULT_TOKENIZER = "/raid/bo/p2pbench/models/Step-3.7-Flash-FP8"
_MAX_OUTPUTS = 50_000
_MAX_PENDING = 10_000
_RT_CHECK_TOKENS = 2048
_THINK_PREFIX = "<think>\n"


def distinct_ngram_ratio(ids: list[int], n: int = 4) -> float | None:
    """Return distinct / total n-grams of ``ids``, or None when shorter than ``n``.

    Args:
        ids: Token ids.
        n: N-gram length.

    Returns:
        The ratio in (0, 1], or None.
    """
    total = len(ids) - n + 1
    if total <= 0:
        return None
    return len({tuple(ids[i : i + n]) for i in range(total)}) / total


def common_prefix_len(a: list[int], b: list[int]) -> int:
    """Return the length of the longest common prefix of two id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class LiveHistoryChatEndpoint(ChatEndpoint):
    """``chat`` endpoint whose assistant history is the server's real output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize as the base endpoint and load the tokenizer now.

        Loading takes ~0.2 s; done lazily it would stall the worker's event
        loop on the first profiled response.
        """
        super().__init__(*args, **kwargs)
        self._prt_tokenizer()

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format as the base endpoint, with tagged assistant messages substituted.

        Args:
            request_info: The request.

        Returns:
            The JSON payload.
        """
        payload = super().format_payload(request_info)
        root = request_info.root_correlation_id or request_info.x_correlation_id
        tagged = subbed = 0
        newest_src = f"{request_info.conversation_id}#{request_info.turn_index - 1}"
        newest_tagged = newest_subbed = False
        isl_delta = 0
        messages = []
        for msg in payload["messages"]:
            src = msg.get(SRC_KEY)
            if src is None:
                messages.append(msg)
                continue
            tagged += 1
            conv, turn, trace_n = src.rsplit("#", 2)
            is_newest = f"{conv}#{turn}" == newest_src
            newest_tagged |= is_newest
            hit = self._prt_lookup((root, conv, int(turn)))
            if hit is None:
                # Copy: ``msg`` aliases the session's raw_messages.
                messages.append({k: v for k, v in msg.items() if k != SRC_KEY})
                continue
            text, n_tokens = self._prt_render(hit, int(trace_n))
            if text is None:
                messages.append({k: v for k, v in msg.items() if k != SRC_KEY})
                continue
            subbed += 1
            newest_subbed |= is_newest
            isl_delta += n_tokens - int(trace_n)
            messages.append({"role": "assistant", "reasoning": "", "content": text})
        payload["messages"] = messages
        payload["return_token_ids"] = True
        pending = self._prt_pending_map()
        pending[request_info.x_request_id] = {
            "tagged": tagged,
            "subbed": subbed,
            "newest_tagged": newest_tagged,
            "newest_subbed": newest_subbed,
            "isl_delta": isl_delta,
        }
        while len(pending) > _MAX_PENDING:
            pending.popitem(last=False)
        return payload

    def process_responses(
        self,
        record: RequestRecord,
        *,
        capture_assistant_turn: bool,
    ) -> tuple[list[ParsedResponse], Turn | None]:
        """Parse as the base endpoint, then store and log this turn's output ids.

        Args:
            record: The completed request record.
            capture_assistant_turn: Passed through.

        Returns:
            The base endpoint's result.
        """
        result = super().process_responses(
            record, capture_assistant_turn=capture_assistant_turn
        )
        info = record.request_info
        if info is None:
            return result
        ids: list[int] = []
        # responses[0] is the role chunk, which carries the (large)
        # prompt_token_ids and no output ids.
        for response in record.responses[1:]:
            obj = response.get_json()
            for choice in (obj or {}).get("choices") or []:
                ids.extend(choice.get("token_ids") or [])
        stats = self._prt_pending_map().pop(info.x_request_id, {})
        root = info.root_correlation_id or info.x_correlation_id
        complete = (
            not record.has_error
            and bool(ids)
            and info.max_tokens is not None
            and len(ids) == info.max_tokens
        )
        store = complete and info.credit_phase == CreditPhase.PROFILING
        rt_lcp = None
        if store:
            # Round-trip check on a bounded prefix (encode costs ~2 us/token on
            # the worker's event loop). "\n" is the gen-prompt token after
            # <think>; it re-tokenizes together with the start of the output.
            head = ids[:_RT_CHECK_TOKENS]
            text = self._prt_tokenizer().decode(head, skip_special_tokens=False)
            reenc = self._prt_encode("\n" + text)
            rt_lcp = max(0, common_prefix_len(reenc, [self._prt_nl_id(), *head]) - 1)
            # The cut can split a token whose bytes re-merge differently.
            rt_n = len(head) - (len(head) < len(ids))
            self._prt_store((root, info.conversation_id, info.turn_index), array("i", ids))
        self._prt_write(
            {
                "x_request_id": info.x_request_id,
                "conv": info.conversation_id,
                "turn": info.turn_index,
                "root": root,
                "phase": str(info.credit_phase),
                "depth": info.agent_depth,
                "max_tokens": info.max_tokens,
                "n_out": len(ids),
                "stored": store,
                "rt_lcp": rt_lcp,
                "rt_checked": None if rt_lcp is None else rt_n,
                "rt_exact": None if rt_lcp is None else rt_lcp >= rt_n,
                "distinct4": distinct_ngram_ratio(ids),
                **stats,
            }
        )
        return result

    def _prt_render(self, out: list[Any], trace_n: int) -> tuple[str | None, int]:
        """Return the assistant content for a stored output and its token count.

        The output is cut to the trace message's length (``<think>`` and the
        ``\n`` after it count) so ISL never grows past the trace's; the cut
        keeps a prefix of the generated ids, which is still a prefix of the
        cached sequence. A shorter output is sent whole. Returns ``(None, 0)``
        when nothing of the output fits.

        Args:
            out: ``[ids, cached (trace_n, text, n_tokens) or None]``.
            trace_n: Token count of the trace message being replaced.

        Returns:
            ``(content, n_tokens)``.
        """
        ids, cached = out
        if cached is not None and cached[0] == trace_n:
            return cached[1], cached[2]
        keep = min(len(ids), trace_n - 2)
        if keep <= 0:
            return None, 0
        text = self._prt_tokenizer().decode(list(ids[:keep]), skip_special_tokens=False)
        # <think> + "\n" + the kept ids, assuming they re-tokenize exactly
        # (rt_lcp in the log says how often they do).
        n_tokens = keep + 2
        out[1] = (trace_n, _THINK_PREFIX + text, n_tokens)
        return out[1][1], n_tokens

    def _prt_store(self, key: tuple[str, str, int], ids: array) -> None:
        """Keep an output locally and publish it to the shared store.

        A session's turns (sub-agent turns after their parent's sticky entry is
        gone) can land on different worker processes, so outputs are also
        written to ``$PRT_LIVE_HISTORY_LOG_DIR/store`` (atomic rename; about
        4 bytes per token). The write finishes before the credit returns, so
        any request that references this turn is issued after it.
        """
        outputs = self._prt_outputs_map()
        outputs[key] = [ids, None]
        while len(outputs) > _MAX_OUTPUTS:
            outputs.popitem(last=False)
        path = self._prt_store_path(key)
        if path is not None:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(ids.tobytes())
            os.replace(tmp, path)

    def _prt_lookup(self, key: tuple[str, str, int]) -> list[Any] | None:
        """Return ``[ids, render cache]`` for an output, from memory or the shared store."""
        outputs = self._prt_outputs_map()
        out = outputs.get(key)
        if out is not None:
            return out
        if not hasattr(self, "_prt_absent"):
            self._prt_absent: set[tuple[str, str, int]] = set()
        # An output is stored before any request that references it is
        # issued, so a key missing now (warmup turn, error) stays missing.
        if key in self._prt_absent:
            return None
        path = self._prt_store_path(key)
        try:
            with open(path, "rb") as f:  # type: ignore[arg-type]
                ids = array("i")
                ids.frombytes(f.read())
        except (FileNotFoundError, TypeError):
            self._prt_absent.add(key)
            return None
        outputs[key] = out = [ids, None]
        return out

    def _prt_store_path(self, key: tuple[str, str, int]) -> str | None:
        log_dir = os.environ.get("PRT_LIVE_HISTORY_LOG_DIR")
        if not log_dir:
            return None
        if not hasattr(self, "_prt_store_dir"):
            self._prt_store_dir = os.path.join(log_dir, "store")
            os.makedirs(self._prt_store_dir, exist_ok=True)
        root, conv, turn = key
        h = hashlib.sha1(f"{root}\0{conv}".encode()).hexdigest()[:24]
        return os.path.join(self._prt_store_dir, f"{h}_{turn}.i32")

    def _prt_outputs_map(self) -> OrderedDict[tuple[str, str, int], list[Any]]:
        if not hasattr(self, "_prt_outputs"):
            self._prt_outputs: OrderedDict[tuple[str, str, int], list[Any]] = OrderedDict()
        return self._prt_outputs

    def _prt_pending_map(self) -> OrderedDict[str, dict[str, Any]]:
        if not hasattr(self, "_prt_pending"):
            self._prt_pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        return self._prt_pending

    def _prt_tokenizer(self) -> Any:
        """Return the raw ``tokenizers`` tokenizer, which is what vLLM encodes with.

        transformers 5's AutoTokenizer maps this checkpoint's LlamaTokenizerFast
        config to a tokenizer that drops whitespace, so it is not used here.
        """
        if not hasattr(self, "_prt_tok"):
            from tokenizers import Tokenizer

            path = os.environ.get("PRT_LIVE_HISTORY_TOKENIZER", _DEFAULT_TOKENIZER)
            self._prt_tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return self._prt_tok

    def _prt_encode(self, text: str) -> list[int]:
        return self._prt_tokenizer().encode(text, add_special_tokens=False).ids

    def _prt_nl_id(self) -> int:
        if not hasattr(self, "_prt_nl"):
            (self._prt_nl,) = self._prt_encode("\n")
        return self._prt_nl

    def _prt_write(self, rec: dict[str, Any]) -> None:
        log_dir = os.environ.get("PRT_LIVE_HISTORY_LOG_DIR")
        if not log_dir:
            return
        if not hasattr(self, "_prt_log"):
            os.makedirs(log_dir, exist_ok=True)
            self._prt_log = open(  # noqa: SIM115 - lives as long as the worker
                os.path.join(log_dir, f"{os.getpid()}.jsonl"), "ab", buffering=0
            )
        self._prt_log.write(orjson.dumps(rec) + b"\n")
