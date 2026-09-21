**Prompt-cache investigation — 2026-09-21**

The reproducible defect is the Copilot SDK session lifecycle. The proxy retained
sessions across tool calls but destroyed them after final answers. Resuming the
same session restored visible history without its encrypted reasoning items.
The next request therefore changed an already evaluated prefix even though
Codex had appended to an otherwise unchanged transcript.

The fix keeps successful sessions connected across user turns. It does not add
cache-key heuristics, rewrite prompts, or special-case Sol.

**Primary evidence and limits**

Source: `~/Downloads/request-trace.jsonl.zip`, modified September 21 at 16:03:44
local time. Archive SHA-256:
`1a0750dc5ca0f4b42e2b622ce91023557b10175544e24fd89d4e2e354b044bef`.
It covers 19:02:55–20:50:07 UTC, with 526 request starts and 527 finishes; the
first finish belongs to a request started before the retained window.

There are 426 non-Excel SDK requests (379 Luna, 47 Sol, including two compact
requests), and 100 Excel requests (67 Luna, 21 Sol, 12 Terra). The non-Excel
traffic is predominantly Codex Desktop 0.155.0-alpha.9.2. There are also small
CLI/manual probes. The Excel control is **Codex traffic routed through the Excel
upstream**, not evidence of a separate Excel client with a different prompt.
The archive contains no direct GHCP Chat Completions sequence and no older,
non-reasoning model sequence. Those paths were checked separately with synthetic
requests; they must not be represented as observed production traffic.

520/526 source and adapter/upstream bodies were truncated to 8,192 bytes.
Complete source item fingerprints survive, as do top-level field fingerprints
and usage. On the SDK route, the field called `upstream_body` is the adapter
input, **not the Copilot runtime's outgoing HTTP body**. Enabling detailed
logging also overwrote the upstream fingerprint summary with that truncated
preview. Consequently, the archive proves source-prefix stability and the timing
of the losses; it cannot independently show every byte of the production
model-wire request or explain the exact cache tier selected by GitHub.

The model-wire difference was reproduced independently using both SDK 1.0.11 /
runtime 1.0.79 (the repository minimum) and SDK 1.0.14 / runtime 1.0.85. The archive
does not record which SDK/runtime version produced it. These replays used a
local synthetic Responses provider, no credentials and no real model calls.

**The concrete request difference**

The real runtime produced this sequence through the proxy adapter:

```text
Request before final answer:
  user(initial prompt)
  reasoning(id=rs_1, encrypted_content=opaque-synthetic-reasoning-1)
  function_call(call_first, inspect, {})
  function_call_output(call_first, Inspection result.)

Next user request, old proxy lifecycle:
  user(initial prompt)
  function_call(call_first, inspect, {})
  function_call_output(call_first, Inspection result.)
  assistant(Done.)
  user(Next question.)

Next user request, fixed lifecycle:
  user(initial prompt)
  reasoning(id=rs_1, encrypted_content=opaque-synthetic-reasoning-1)
  function_call(call_first, inspect, {})
  function_call_output(call_first, Inspection result.)
  reasoning(id=rs_2, encrypted_content=opaque-synthetic-reasoning-2)
  assistant(Done.)
  user(Next question.)
```

The first meaningful divergence in the reproduction is **input[1]**: the old
request has a function call where the cached request had encrypted reasoning.
Both prior reasoning items disappear. Model, tools, instructions, reasoning
configuration, text configuration and prompt-cache key remain equal. The
runtime also regenerates assistant message IDs; the replay comparison explicitly
ignores only those IDs, preserving reasoning IDs, call IDs and all content.

`_release_session` caused `_evict_live_session` → `session.disconnect()` → SDK
`session.destroy`. `_open_session` subsequently called `resume_session` for the
same ID. The runtime's disk restoration lost the encrypted history. Codex did
not request that rewrite. `_render_input_segments` ignores client reasoning
items, and the adapter returns readable summaries rather than the runtime's
encrypted blobs, so the next caller transcript cannot repair the lost SDK state.

This explains why preserving the session ID alone was insufficient, and why
the long tool loops looked good while the next user turn collapsed.

Current OpenAI documentation says GPT-5.6 defaults to persisted reasoning across
turns. Stateless continuation requires replaying the response items, including
encrypted reasoning; readable summaries are not equivalent. Earlier families
can omit prior-turn reasoning under `current_turn` semantics.
[Reasoning documentation](https://developers.openai.com/api/docs/guides/reasoning).

**Sol sequence from the trace**

Conversation-key fingerprint `7ef188ab4b11d359`, model `gpt-5.6-sol`, effort
`high`. Every prior source item is byte-content-identical under the recorded
canonical hashes across all 32 requests. Instructions, tools and their order,
model, reasoning, verbosity/text, includes, store and stream settings remain
stable. Only client metadata changes at new user turns.

Input tokens below are reported model usage. “Prior input candidate” is the
previous request's total input, showing the amount of history that should remain
eligible if preserved. It is **not a guaranteed exact cached-token count**:
eligible boundaries, hidden context, provider accounting and tokenization prevent
deriving that exact count from these logs. Unchanged items are exact source
fingerprint comparisons. Each normal continuation appends output/tool results;
those additions should not invalidate earlier boundaries.

`D*` means a new user turn after a final answer: the old proxy takes the
destroy/resume path. Its reasoning-removal effect is established by code and the
runtime reproduction above, rather than a captured production wire body. The
first source difference remains the appended suffix, not a rewritten old item.

| UTC / request | Input tokens | Prior input candidate | Cached tokens | Unchanged source items | First break |
|---|---:|---:|---:|---:|---|
| 20:35:53 / `3c0caa38` | 5,433 | — | 0 | — | Initial request |
| 20:36:01 / `989cc98e` | 7,448 | 5,433 | 5,120 | 3/3 | Appended suffix |
| 20:36:09 / `4becaf03` | 7,713 | 7,448 | 7,168 | 6/6 | Appended suffix |
| 20:36:24 / `342b9da5` | 7,596 | 7,713 | 5,120 | 8/8 | D* |
| 20:36:37 / `b020cfc3` | 8,049 | 7,596 | 7,168 | 10/10 | Appended suffix |
| 20:36:44 / `e3e4d35c` | 12,600 | 8,049 | 7,680 | 13/13 | Appended suffix |
| 20:37:08 / `f8a81bd6` | 15,076 | 12,600 | 12,288 | 18/18 | Appended suffix |
| 20:37:20 / `a1bbc33d` | 20,047 | 15,076 | 14,848 | 25/25 | Appended suffix |
| 20:37:43 / `a948892a` | 25,814 | 20,047 | 19,456 | 31/31 | Appended suffix |
| 20:37:57 / `3ed89be2` | 27,103 | 25,814 | 25,600 | 38/38 | Appended suffix |
| 20:38:01 / `05ab3b14` | 28,137 | 27,103 | 26,624 | 41/41 | Appended suffix |
| 20:38:07 / `27e5687c` | 28,523 | 28,137 | 27,648 | 46/46 | Appended suffix |
| 20:38:15 / `439f206b` | 30,142 | 28,523 | 28,160 | 49/49 | Appended suffix |
| 20:38:25 / `c68430b9` | 30,816 | 30,142 | 29,696 | 51/51 | Appended suffix |
| 20:38:30 / `1fface3d` | 31,757 | 30,816 | 30,720 | 54/54 | Appended suffix |
| 20:38:34 / `740101fd` | 32,120 | 31,757 | 31,232 | 56/56 | Appended suffix |
| 20:38:44 / `b2338820` | 33,231 | 32,120 | 31,744 | 58/58 | Appended suffix |
| 20:38:53 / `f62e59ac` | 33,592 | 33,231 | 32,768 | 61/61 | Appended suffix |
| 20:39:40 / `0ca434e5` | 30,273 | 33,592 | 0 | 64/64 | D* |
| 20:40:45 / `71f91461` | 30,741 | 30,273 | 7,168 | 66/66 | D* |
| 20:41:02 / `6e1e4cea` | 31,709 | 30,741 | 30,208 | 68/68 | Appended suffix |
| 20:41:08 / `52602755` | 32,946 | 31,709 | 31,232 | 71/71 | Appended suffix |
| 20:41:17 / `a3dc8d12` | 33,088 | 32,946 | 32,768 | 74/74 | Appended suffix |
| 20:41:24 / `8c897d41` | 33,245 | 33,088 | 32,768 | 77/77 | Appended suffix |
| 20:41:26 / `7a040f52` | 34,087 | 33,245 | 32,768 | 79/79 | Appended suffix |
| 20:41:39 / `d677a2d5` | 35,142 | 34,087 | 33,792 | 81/81 | Appended suffix |
| 20:41:45 / `6165a591` | 35,286 | 35,142 | 34,816 | 84/84 | Appended suffix |
| 20:41:52 / `09e68914` | 35,567 | 35,286 | 34,816 | 87/87 | Appended suffix |
| 20:42:01 / `9408ed12` | 35,770 | 35,567 | 35,328 | 89/89 | Appended suffix |
| 20:42:11 / `041b3def` | 36,051 | 35,770 | 35,328 | 91/91 | Appended suffix |
| 20:42:14 / `10060473` | 36,305 | 36,051 | 35,840 | 93/93 | Appended suffix |
| 20:42:26 / `d188fcaa` | 37,011 | 36,305 | 35,840 | 95/95 | Appended suffix |


The strongest boundary is `f62e59ac` → `0ca434e5`: source input grows from 64 to
66 items (217,515 → 218,145 serialized trace bytes), all 64 historical item
hashes match, yet upstream input shrinks from 33,592 to 30,273 and cached input
falls from 32,768 to zero. The preceding user turn accumulated 3,350 reported
reasoning tokens. The 3,319-token shrink despite a new visible suffix is
consistent with losing that reasoning. A second user turn, `71f91461`, keeps
all 66 old source items but only reports 7,168 cached tokens.

An independent earlier Sol sequence shows the same pattern: `81c1db3a` at
19:08:58 has 11,750 input / 11,264 cached; `5197bb9b` at 19:18:09 preserves all
39 previous source items but has 11,220 input / 6,656 cached. Luna also shows
the defect: `503d0868` at 19:36:43 has 152,724 input / 148,893 cached; the next
user turn `f9ea3d94` at 19:43:35 preserves all 340 source items but has 141,218
input / 1,548 cached.

The archive cannot prove why one damaged prefix hits a short cache boundary and
another reports exactly zero. It establishes the invalidation pattern, not all
upstream routing/retention decisions. No claim here treats cached_tokens alone
as proof of prefix preservation.

**Excel control**

These consecutive Sol/Excel requests use conversation-key fingerprint
`8fe8178bc11eb402`. The first follows a Terra-to-Sol model change, so it is not
an appropriate warm same-model baseline. Every later row preserves every prior
source item. The Excel adapter preserves encrypted reasoning and constructs a
stable developer/tool-catalog prologue before the growing history. It has no
SDK destroy/resume transition. This prefix property was also verified with a
synthetic final-answer/new-user sequence through `prepare_responses_body`.

| UTC / request | Input tokens | Prior input candidate | Cached tokens | Unchanged source items | First source difference |
|---|---:|---:|---:|---:|---|
| 19:19:48 / `16967b5c` | 67,374 | Different model | 8,820 | — | Model switch |
| 19:19:58 / `ef92b30f` | 67,831 | 67,374 | 67,188 | 56/56 | Appended suffix |
| 19:20:07 / `8e6fe7a8` | 68,870 | 67,831 | 67,188 | 60/60 | Appended suffix |
| 19:20:18 / `523c4fbc` | 72,754 | 68,870 | 67,188 | 63/63 | Appended suffix |
| 19:21:13 / `7f9a2de7` | 75,956 | 72,754 | 72,564 | 66/66 | Appended suffix |
| 19:21:18 / `bd200d1f` | 77,939 | 75,956 | 75,764 | 71/71 | Appended suffix |
| 19:21:26 / `eda157c5` | 79,204 | 77,939 | 77,812 | 74/74 | Appended suffix |
| 19:21:36 / `44a59f62` | 80,008 | 79,204 | 79,092 | 78/78 | Appended suffix |

Excel is a useful control, not a guarantee of a hit on every request: the Terra
sequence also has an isolated cache dip with stable source fingerprints. The
truncated upstream record cannot establish its exact cause.

**Audit of other potential prefix changes**

| Area | Evidence / disposition |
|---|---|
| Message insertion/removal/order | Selected Sol source histories append only. SDK disk resume deletes reasoning; this is the reproduced defect. |
| System/developer reconstruction and role conversion | SDK initial input is rendered into one user prompt, with top-level instructions passed as system configuration. Subsequent live requests send only new caller text. Excel has a stable prologue. Existing architectural differences, not a reason to sort/rewrite historical input. |
| Tool definitions, ordering, MCP/custom schemas | Source tools fingerprints stay constant in the selected sequence. SDK registers function/custom declarations in source order and skips unsupported tool types. No evidence that changing tool schemas caused these boundaries. Runtime replay confirms identical wire tools across the break. |
| JSON key ordering | Trace hashes canonicalize object keys; wire JSON bytes are not tokenized verbatim as a chat prompt. No global key sorting was added. Arguments now have their own diagnostic hash. |
| Model aliases/remapping | Requested/resolved model names are equal in the failing Sol sequence. Excel strips its provider suffix deliberately. Runtime replay verifies the actual outgoing model field. Cross-model Excel switch excluded from warm baseline. |
| Effort, verbosity and output configuration | Recorded fingerprints remain fixed at failure boundaries; replay verifies stable runtime configuration. Changing configuration can legitimately require a reconnect; no effort downgrade added. |
| Conversation/cache IDs, continuation and previous response IDs | The caller cache-key hash stays fixed. SDK continuation call IDs encode its session and pending request. The failing sequence contains no previous_response_id. Reproduced disk resume uses the same SDK ID and wire cache key, yet still loses reasoning. |
| Codex metadata | New turn metadata changes, historical source item hashes do not. Metadata is not included by SDK transcript rendering. Stable thread identity is used for session aliases. |
| GHCP metadata | SDK owns authentication, routing and its model request metadata. Original trace does not capture those wire fields. Direct GHCP Chat Completions requests are absent and were checked independently with synthetic endpoint calls. |
| Timestamps, paths, environment and Git state | Caller environment content is stable in the selected sequence. Real-runtime replay adds current_datetime to each new user message and retains older timestamps; that append does not rewrite the prefix. Workspace options are unchanged during a conversation. No speculative removal of environment/Git context. |
| Compaction/truncation | Selected 32-request Sol sequence has no caller compaction. Separately, Luna input falls from 174,298 to 28,464 at 19:32:05 while source history continues growing: SDK context management is distinct from source history. Do not classify all size drops as this defect. |
| Content arrays/strings and multimodal conversion | SDK renderer joins text, represents images as placeholders, and ignores client encrypted reasoning. Those are existing SDK-adapter limitations. Native Responses/Excel preserve their own history; no new flattening or normalization introduced. |
| Cache controls / retention | Selected requests supply prompt_cache_key, not explicit breakpoints/retention. SDK owns its generated key. No cache controls are being added to compensate for a history deletion. |
| Token accounting | Source JSON bytes, visible chars, encrypted bytes, and upstream input tokens are different quantities. SDK traces can contain aggregate per-call usage, and old displayed total_tokens excluded reads. All tables use input_tokens/cached_input_tokens, not displayed totals. |
| Responses versus Chat Completions | SDK is selected only by the Responses route. The Chat Completions route forwards messages directly; synthetic Sol and GPT-4.1 requests confirm prefix/tool preservation there. It does not execute this SDK session lifecycle. |

GPT-5.6+ caching checks eligible message boundaries rather than promising reuse
of every arbitrary token prefix. Deleting reasoning changes every later prefix;
adding a suffix preserves existing boundaries. Tool schemas and prefix settings
must stay stable. Explicit breakpoints cannot restore deleted history. Cache
retention and prefix eligibility are separate from local session lifetime.
[Current prompt-cache documentation](https://developers.openai.com/api/docs/guides/prompt-caching).

**Review of today's changes**

| Commit | Assessment |
|---|---|
| `8bd16b0` — Preserve Codex prompt cache replay prefix | Removes internal_chat_message_metadata_passthrough. Reasonable sanitation for native forwarding, but ineffective for this failure: SDK rendering already ignores that envelope. Some long traces have item_hash-only changes while content/output hashes and cache reuse continue normally. Kept; not presented as the solution. |
| `4a8e21b` — Bound Responses function item IDs | Addresses provider ID-length limits. Does not change the session lifecycle. Kept. |
| `1d19bb0` — Excel run_officejs transport | Excel-specific call/result handling. Working control stays on that path. Kept. |
| `851ba89`, `6033d72`, `097beca` — Reasoning/subagent streaming | Presentation and completion of readable summaries; do not persist encrypted SDK reasoning across destroy/resume. Kept. |
| `0aadfd4` — Workspace Git context and token totals | Corrects workspace context and gross Responses total_tokens. A changed option can cause a one-time reconnect; stable options do not explain repeated end-of-turn destruction. Correct totals can make context consumption more visible without causing a cache miss. No evidence justifies reverting these changes. |

Earlier September 17 changes disabled cache-settle delays and corrected
Excel session identity/billing. A sleep, affinity header or billing adjustment
cannot recover items removed during SDK restoration. The faulty release
condition predated today's patches; those patches did not replace it.

**Implemented changes and verification**

`copilot_sdk_upstream.py` now retains successful final-answer sessions as well as
tool continuations. Default idle retention is 1,800 seconds, configurable through
the existing environment setting. A capacity of 32 completed idle sessions
bounds retained runtime state; active work, pending calls and active compaction
are excluded from that capacity eviction. Failures still disconnect. Tests cover
the final-answer boundary, expiry, capacity, live reuse, compaction and explicit
configuration changes. Trace completion records distinguish create, reuse_live,
resume_disk and recovery fallbacks, with reasons for live-reuse misses.

`proxy.py` keeps full upstream item fingerprints beside capped debug previews,
adds argument hashes, and labels SDK adapter bodies so they cannot be mistaken
for model-wire captures. A regression test verifies late-item fingerprints remain
available beyond the 8 KB preview. No Excel translation code changed.

`tools/analyze-prompt-cache-trace.py` produces all 526 request comparisons without
printing prompt text. `tools/verify-sdk-cache-continuity.py` drives the real SDK
runtime through the proxy's session open/dispatch/release path against intercepted
synthetic Responses, including an external tool round-trip and a new user turn.

| Runtime replay | Old lifecycle | Fixed lifecycle |
|---|---|---|
| Sol, encrypted reasoning | 1/4 prior input items retained as a prefix; 0 reasoning items | 4/4 retained; both reasoning items retained |
| Luna, encrypted reasoning | 1/4; 0 reasoning items | 4/4; both reasoning items |
| GPT-4.1, no reasoning | 3/3 prior items | 3/3 prior items |
| Excel final-answer/new-user transformation | Preserves encrypted prefix | Unchanged; preserves encrypted prefix |

These are **request-structure results, not measured improvements in live cache
hit rates**. Synthetic usage values are deliberately not used as evidence.
The runtime probe also shows why models with no reasoning are less affected by
this particular deletion. Model requests are never sent to OpenAI/GitHub during
the probe, although the SDK can download its runtime on first use.

Validation completed: 146 focused unittest checks, 28 targeted Copilot session
semantics checks, syntax compilation, and `git diff --check`. Runtime replays
passed with both SDK/runtime pairs listed above. Authenticated client replacement
also clears the retained pool so a dead runtime wrapper cannot be reused.

Reproduce with the repository virtualenv (Windows):

```powershell
.\.venv\Scripts\python.exe tools\analyze-prompt-cache-trace.py "$HOME\Downloads\request-trace.jsonl.zip"
.\.venv\Scripts\python.exe tools\verify-sdk-cache-continuity.py --output .cache\cache-investigation\replay
.\.venv\Scripts\python.exe -m unittest test_copilot_sdk_upstream test_excel_upstream test_reasoning_translation test_responses_replay_ids test_request_prompt_archive -q
```

**Remaining limits**

Cache expiration, genuine model/settings/tool changes, user edits/forks and
compaction can invalidate earlier prefixes. GPT-5.6 message-boundary lookup
also limits which otherwise stable prefixes are reusable. The SDK still has
lossy disk persistence: process restart, expiry, capacity eviction, failures or
configuration changes that require reconnecting can lose encrypted reasoning.
That is a separate SDK limitation, not an unavoidable Codex requirement. This
patch removes the gratuitous destruction after every answer; it does not claim
to make disk restoration lossless. The SDK route also remains an agent-session
adapter, not a complete native Responses passthrough.

A live post-deployment trace is needed to quantify provider cache-hit gains and
investigate any residual provider-specific misses. No production deployment or
paid upstream replay was performed during this investigation.
