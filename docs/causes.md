# Cause classes

Each request is compared with the closest of the last 8 prompts for the same model. The prompt is
serialized as system messages, then tools, then history; offsets are characters into that form,
reported as `<part>:<offset>` such as `sys[0]:312`. Rules run in this order; the first match wins.

## tool_order

The same tools arrived in a different order. qwen-code built the tool list from an unordered map
(qwen-code #8277). Fix at the source: emit tools in a deterministic order.

## history_truncated

The oldest turn was dropped, so every message after the system prompt moved. Fix at the source:
truncate or summarize once, at a boundary, instead of sliding the window every request
(smfclearinghouse prefix-caching guide).

## history_summarized

Older turns were folded into one summary message, so everything after that message moved. Aider
does this in the background once the history passes a token limit ("I spoke to you previously
about a number of things…"). Detected when the message where the prompts diverge, and the one after
it, are both gone and the history got shorter. Fix at the source: summarize rarely and in large
steps; in Aider raise `--max-chat-history-tokens`.

## block_after_history

The client re-sends a block after the history and inserts each new turn before it, so the block and
everything after it are processed again every turn. Aider sends the current contents of the files in
the chat this way. Detected when the message that used to sit at the divergence point now appears
later in the prompt, possibly with new content. Fix at the source: send the re-sent block before the
history. 11 of 27 misses in a 29-turn Aider run (qwen3:4b, Ollama) were this cause.

## whitespace

The differing part is equal once whitespace is ignored, for example `- be brief` versus
`-  be brief`. Fix at the source: serialize messages with stable whitespace and JSON formatting.

## uuid, iso_datetime, unix_epoch, hex_nonce

A volatile token covers the first differing character. Hermes Agent inserted `Current time:` after
context compaction (hermes-agent #8687, fixed by a date-only line in PR #27675). Claude Code puts an
attribution block with a per-request hash at the top of the system prompt; set
`CLAUDE_CODE_ATTRIBUTION_HEADER=0` (claude-code-router PR #1220). Fix at the source: date-only lines,
or move volatile values to the end of the prompt.

## system_replaced

The system prompt differs early and its end differs too: a different prompt, typically a side
request such as title generation. On single-slot backends it evicts the main conversation's cache
(hermes-agent #8687). Fix at the source: send side requests to a separate slot or model.

## content_changed

Anything else that differs, such as an edited earlier message. Aider rewrites two messages in place:
its repo map, refreshed after edits (11 of 27 misses in the Aider run), and its history summary once
it exists. Fix at the source: keep sent history immutable; in Aider try `--map-refresh manual` and a
higher `--max-chat-history-tokens`.

## reprocessed

The prompt extends the previous one, yet llama.cpp measured zero reused tokens. Hybrid and
sliding-window models force full re-processing in llama.cpp regardless of prefix
(https://github.com/ggml-org/llama.cpp/issues/24714); otherwise the slot was evicted. A proxy cannot
fix this; it reports it.

## Not misses

`new_prompt` (nothing similar seen yet), `prefix_match` (the prompt extends a recent one),
`unparsed` (not a chat request; forwarded unchanged) and `skipped_large` (over 2 MB; forwarded
without inspection).
