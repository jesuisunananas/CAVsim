# Buffer-based cross-channel synchronization

**Date:** 2026-08-18
**Scope:** `co-perception/src/co_perception/ingest/frame_sources.py`, `co-perception/scripts/process_video.py`, `co-perception/src/co_perception/config.py`, `co-perception/config/pipeline.yaml`

## What this replaces

The previous design (`LocalSocketSource`) kept only a single "latest frame" slot per channel — a new arrival always overwrote whatever was there. Cross-channel alignment was done by comparing whichever frame *happened* to be sitting in each channel's slot at the moment of a tick, using a median-based tolerance check, and **silently processing whichever subset of channels currently agreed**, excluding stragglers rather than ever requiring all 4 together.

That was wrong per explicit instruction: "process 4 channels together with the same timestamp... you must only process the channels if they are aligned and synchronized." A single-slot design can't actually do this — once a faster channel's older frame is overwritten by a newer one, there's nothing left to match a lagging channel against, even if the lagging channel's data would otherwise still be recoverable.

## What was built

**`LocalSocketSource` now buffers, not overwrites.** Each channel keeps a bounded FIFO (`collections.deque`, sized via the new `ingestion.sync_buffer_seconds` config field × `nominal_fps`) holding recent frames in arrival order, instead of a single slot. This is what makes "hold the fast channels' data until the slow one catches up" possible at all.

**Synchronization now requires all 4 channels to genuinely match before anything is processed.** New methods on `LocalSocketSource`:
- `peek_newest()` — the most recently arrived frame (drives the sync target, explained below)
- `find_closest(target_t, tolerance)` — non-destructive search through the buffer for the frame closest to a target timestamp
- `discard_through(index)` — the commit step, only called after *all 4* channels have confirmed a match, so a match on one channel never mutates its buffer if another channel turns out to have no match at all

## A real bug found and fixed before this shipped

The first version of the new algorithm compared each channel's **oldest unconsumed** frame (the natural-seeming choice: "process in timestamp order"). I validated it with a synthetic test simulating the exact scenario from our discussion (channel 3 stalled 15 seconds behind, channels 0-2 healthy and agreeing with each other) — it passed.

But that test was too easy: it had 3 channels already agreeing with each other from the start, so it never exercised the case where **all 4 channels have a persistent mutual offset from each other simultaneously** — which is what the real pipeline actually looks like (each channel's decode process runs its own independent GOP-drop cycle, so they're each 1-2 seconds off from each other more or less permanently, not just one straggler among agreeing peers).

I caught this by watching the real buffers directly before trusting the design further: connected `LocalSocketSource` to the live decode sockets and printed the oldest-buffered timestamp per channel over 15 seconds. It never moved — spread stayed frozen at ~1.08s the entire time, only "resolving" once a buffer completely filled and started auto-evicting, which isn't real synchronization, just an overflow artifact. Deployed against the real pipeline, this produced **zero output for 20+ seconds** — a genuine deadlock: a channel's oldest-buffered item only advances when something pops it, and nothing pops until aligned, so if no two channels ever naturally agree on their own, nothing ever moves.

**The fix:** key the sync target off each channel's *newest* arrival instead (`target_t = min(newest across all 4)`), not the oldest. Unlike the oldest/front item, the newest item always advances as data arrives, regardless of whether anything's been consumed — so `target_t` makes real forward progress on its own. Every channel (including whichever is currently the laggard) then searches its own buffered history for the frame closest to `target_t`.

## Testing methodology

Given two design iterations already had real, distinct bugs, I validated the final version with synthetic tests before deploying, then re-validated directly against the live pipeline before calling it done — not just "it compiles, ship it."

**Synthetic tests** (`/tmp/.../test_sync_algo3.py`, using mock sources with the same buffer interface, no camera pipeline involved):
1. All 4 channels healthy and identical, gradual 30fps arrival, `target_fps=2` → 20 sets over 10s, exactly 2.00 sets/sec, all internally aligned.
2. Channel 3 stalled 15s then recovers → 10 sets processed while stalled (all correctly bounded below t=15.5, never jumping ahead of the laggard), continues correctly once it catches up, reaching t=19.5 of ~20s available.
3. **The deadlock case**: all 4 channels persistently offset from each other (+1.0s, 0.0s, -0.5s, +0.3s, none agreeing, none transiently stalled) → 37 sets processed at ~1.85 sets/sec, no deadlock, all aligned.
4. Same as (3) plus ±5ms random per-frame jitter (smaller than the 50ms tolerance) → 28 sets processed, all aligned.

An earlier attempt at test 1 gave a misleading false failure (1 set instead of 20) because the test harness itself pushed all frames instantly before running the algorithm, letting it see the "newest" timestamp jump straight to the far future — something that can't happen with real gradual arrival. Rewrote the harness to interleave pushing and syncing one frame at a time before trusting the results.

**Live verification**, after deploying to the real pipeline (`target_fps=2`, `sync_buffer_seconds=8`):

| Metric | Result |
|---|---|
| All 4 channels represented in WS output (20s sample) | ch0: 30, ch1: 29, ch2: 29, ch3: 29 — even, not lopsided |
| Per-channel achieved rate | 1.45-1.50 fps (target was 2 — see caveat below) |
| Decode lag (sanity check — should be unaffected) | 5.09-5.64s, spread 0.55s — consistent with pre-existing decode state, confirming this change didn't touch decode |
| Co-perception RSS | 6.1GB — consistent with the new buffering (8s × 4 channels of history), not runaway |

Since the algorithm has no code path that emits a processed set without all 4 channels having matched within tolerance, this even 4-way distribution is itself evidence of correct alignment — the previous median-based design would have shown exactly this kind of lopsided/missing-channel pattern (and did, when tested right before this fix, per the last session's numbers: ch3 didn't appear at all in a 20s window, ch0/ch1/ch2 were uneven).

## Honest caveats

- **Achieved rate (1.45-1.50fps) is somewhat below the configured target (2fps).** This is expected given the tolerance/`target_fps` interaction and decode's own residual jitter — not every target_fps-spaced instant has an achievable 4-way match within 50ms, so some are skipped rather than forced. Not investigated further today; flagging rather than claiming it's exactly 2.0.
- **Memory scales with `sync_buffer_seconds`.** Worth remembering if that value is ever increased — cost is real (frames at this resolution are ~14MB each) and was called out directly in both the code and config comments.
- This was deployed and spot-checked over roughly a minute of live operation, not a long soak — the earlier decode work (GOP-drop, CPU reservation) both looked fine on initial deploy and revealed real problems only after sustained observation, so the same caution applies here.
