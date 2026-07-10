# A-DRAFTAUX: sibling-differential draft auxiliary objective (design)

## Motivation (evidence-backed)
Critic prices sibling gates at battle start (sign 95-100% from 1.5M steps;
|dV| 0.002-0.005) but pick-head PPO never converts it (KL == 0 through 45M x
all levers). GAE already delivers battle-start V undamped to the last pick —
the bottleneck is that the GATE-DIFFERENTIAL component of V is ~10x smaller
than V's estimation noise at pick steps. Fix: inject the differential
directly as a dense pick-trajectory signal.

## Mechanism
At each draft->battle boundary (deck_context.mode transitions !=0 -> 0):
  V_own  = critic value at the real battle-start obs (already in self.values)
  V_sib  = critic value at the SAME obs with ONLY the own-gate identity swapped
           to its sibling (deck_context.gate_card_def_id + gate zone card id),
           computed with the CORRECT LSTM state at that step
  r_aux  = AZK_DRAFT_SIBDIFF_COEF * clamp(V_own - V_sib, -CAP, CAP)   [detached]
Injected into self.rewards at the LAST PICK step (the step whose next obs is
the boundary) for the drafting agent row. gamma-chain propagates it back
through the pick trajectory. Optionally annealed with the shaping scale.

## Implementation plan (trainer.py, post-rollout pre-train hook)
1. Boundary detection: view self.obs uint8 buffer as NATIVE_DECKBUILD dtype;
   mode field per (step, row); boundary where mode[t]==0 and mode[t-1]!=0.
   (native path only; guard on deck_building config.)
2. LSTM state at boundary: stored per-bptt-segment initial (h,c) in the
   buffers (as used by the learn pass). Unroll encode+cell over the segment
   containing the boundary (<=bptt_horizon steps, batched over boundary rows
   only) to reach state_{t-1}.
3. V_sib: encode(swapped_obs_t) -> cell(state_{t-1}) -> value head. Swapped
   obs built by byte-copy + two field writes (deck_context gate id, gate zone
   slot). Sibling map from the catalog (same as env knob).
4. Parity check: V_own recomputed this way must match self.values[t] within
   bf16 tolerance (validates state reconstruction) — assert in a test.
5. Inject: self.rewards[t_lastpick, row] += coef * (V_own - V_sib). t_lastpick
   = boundary step - 1 in the same row's stream (careful with segment edges:
   if the boundary is the first step of a segment, the last pick lives at the
   end of the previous segment — index in the flat (time, agent) buffers).
6. Knobs: AZK_DRAFT_SIBDIFF_COEF (0 = off, default), AZK_DRAFT_SIBDIFF_CAP
   (default 0.05), multiply by current shaping anneal scale.
7. Cost: boundaries per epoch ~ episodes/epoch (~50-100 rows) x 16-step unroll
   — negligible vs the train pass.

## Validation ladder
a. Unit: parity of reconstructed V_own vs stored values (4).
b. Unit: swapped-obs V_sib == probe_critic_gate's measurement on the same
   checkpoint/state (reuse probe machinery for one fixed case).
c. Smoke arm draftaux1 (15M, portalgp recipe + coef 0.02): watch
   losses/policy health, draftref, and KL trajectory. GO metric: KL > 1e-3 at
   any checkpoint (100x every measurement to date) without draftref
   regression > 3pp.

## Risks
- Reward injection biases the value target for the last pick (V trains toward
  V_own + aux) — small coef + cap + anneal keeps it a perturbation.
- CUDA-graph rollout unaffected (all work is post-rollout, out of capture).
- If bptt segment states aren't stored per segment in this trainer, fall back
  to zero-state unroll from episode start of the draft (102 steps, still cheap).

## FINAL EXPERIMENT MATRIX (converged with user, 2026-07-09 20:20)
All on the portalgp production base (portalgp1 @15M 46.9% = free control):
  auxv1  : aux1 only  — boundary V-bootstrap reward, slow-annealed coef
  auxd1  : aux2 only  — CLIPPED sibling differential max(0, V_own - V_sib),
           shaping-annealed coef (never run unclipped/alone: the differential
           can be gamed by making the deck WORSE under the sibling)
  auxvd1 : both
15M each (~2h GPU). Readouts: draftref 96, gate-swap KL (headline — any KL>0
is a first), critic ratio, sibling composition divergence. Winner (KL or
composition moves, external regression <= 3pp) -> 45M with full trajectory
suite. If none move: credit-path length and signal amplification are both
excluded -> distributed scale is the last remaining hypothesis.
Knobs: AZK_DRAFT_VBOOT_COEF, AZK_DRAFT_SIBDIFF_COEF, AZK_DRAFT_SIBDIFF_CAP.
