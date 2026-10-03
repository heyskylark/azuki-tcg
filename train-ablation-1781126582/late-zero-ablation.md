# Late Shaped-Reward Removal Ablation

Status: completed; late zero is viable-neutral, not adopted (2026-07-21)

## Question

Can the accepted mature policy remove all shaped reward late in training without
losing terminal strength, deck quality, or learned battle mechanics?

This is not a repeat of `shaped_reward_tail45_v1`. That treatment began reducing
shaping at p6000 and spent 900 of 2,930 continuation updates (30.7%) at zero. It
finished materially weaker than its fixed-floor control: -8.9 paired points on
the p4870 panel, -4.5 points on heldout references, -9.7 heldout Water points,
fewer attacks and spells, and longer games. Those effects are much larger than
the approximately one-point heldout movement that this project treats as noise.

The remaining hypothesis is narrower: first mature one shared policy for most
of the continuation, then remove the final `0.15` floor later and leave exactly
the final 20-40% of training at zero.

## Atomic Parent And Reward Stack

Both arms ultimately descend from the last accepted atomic parent, p4870:

- model SHA-256: `def350888ecbf9014e590fb8540080cb26b6c14852882caa85def3f6db8a4f76`
- trainer SHA-256: `ce1aa27a1ab58e4c9060f6de99b86cdfc6e8fe815803d9b3ba04155848aa3486`
- league SHA-256: `22b247ba70c6d6a72f92ab4af1e805e874c2aa5337a4bba5ea165145f3e236de`
- promotion-state SHA-256: `9a1557ee90dbaa85cd40889c411be73b6ba8a7d6e169dad139d804fc413a7b0a`
- completed episodes: `404`

The shared trunk retains the complete accepted p4870 recipe: leader delta
`1.25`, board delta `0.35`, no-op penalty `0.02`, truncation edge `0.45`,
untapped IKZ `0.15`, portal GP `0.3`, tempo-deduplicated early tempo `0.1`,
damage mitigation `0.15`, temporary charge `0.08`, and temporary attack
`0.025` per damage capped at four. Native episode annealing remains at its
mature `0.15` floor. Unaccepted entity exchange, IKZ conversion, contextual
reserve, draft auxiliary credit, leader credit, and whole-draft credit remain
zero. XGate masking and PFSP remain enabled. Reference-deck league seats remain
disabled. Promotion stays shadow-only and cannot affect the opponent pool.

## Matched Schedule

The continuation ends at p7800 and contains 2,930 updates / 45,004,800 learner
rows. It restarts one p4870-p7800 cosine LR schedule at peak `0.00003`; no
branch or process restart may restart that schedule.

1. **Shared trunk, p4871-p6900:** train once at effective shaping scale `0.15`.
2. **Exact fork at p6900:** copy the model, trainer/optimizer, completed-episode
   count, league state, promotion state, and opponent directory into both arms.
3. **Ramp, p6901-p7200:** control remains at `0.15`; treatment linearly reduces
   the trainer multiplier from `1.0` at p6900 to `0.0` at p7200.
4. **Zero tail, p7200-p7800:** treatment multiplier remains exactly zero while
   true terminal reward and win labels remain active. Control remains at
   effective scale `0.15`.

Counting p7200 through p7800, the treatment has 601 zero-multiplier updates,
20.51% of the p4871-p7800 continuation. This satisfies the requested 20-40%
zero interval while delaying removal by 1,200 updates relative to the rejected
schedule. Both branches restart once at p7200 to prove schedule persistence.

The fork must be byte-identical across arms. Existing old p6900 artifacts are
not eligible because their corresponding league state later advanced to p7800.

## Fork Qualification

Before branches train, evaluate the fresh shared p6900 checkpoint against p4870
for 192 fixed-seed, seat-paired games. Continue only with zero timeouts, score
at least `0.42`, and 80% paired lower bound at least `0.39`. This is a permissive
non-collapse check, not model promotion and not evidence for either arm.

## Evaluation

The primary outcome is fixed-panel terminal performance, not shaped training
`episode_return`. Returns before and after multiplier removal have different
units and cannot demonstrate ongoing improvement.

Run 192-game same-epoch treatment-versus-control panels at p7200, p7300, p7400,
p7500, p7600, p7700, and p7800. Report endpoint score, the mean of the final
three panels, a robust score slope across the zero interval, and min/max
oscillation. Oscillation is acceptable; a falling late mean or mechanics
collapse is not.

At p7200, p7500, and p7800, evaluate both arms with:

- 192 paired games against p4870;
- 288 heldout-reference games, plus the training-reference panel at p7800;
- 16 sampled drafts per gate plus greedy deck composition;
- 64 CPU self-play games with legal-opportunity logging.

Report portal and unique-card counts, four-of share, average cost and card-type
mix, Water spell slots and proactive/response use per legal window, direct
Garden choice as a diagnostic, attacks, spells, portals, no-ops, and game
length. Sibling-portal KL is omitted because prior measurements were repeatedly
near zero and it is not the question under test. Promotion is telemetry only.

## Decision Rules

Integrity requires byte-identical fork artifacts, exact absolute-update
multipliers, native scale `0.15`, active terminal labels after p7200, zero
terminal/draft credit, exact LR continuation, and zero evaluation timeouts.
Median SPS in each arm must remain at least `1235`, and treatment must retain at
least 95% of control SPS.

Treat an approximately one-point heldout delta as neutral. Do not choose an arm
from that signal alone. Evidence must be coherent across direct H2H, paired
parent results, the late direct-score trajectory, and mechanics.

- **Adopt zero:** endpoint direct score at least `0.52`, 80% lower bound at
  least `0.48`, final-three mean at least `0.51`, nonnegative robust late slope,
  and all safety gates pass.
- **Zero viable but neutral:** no material terminal or mechanics regression,
  endpoint direct score at least `0.48`, final-three mean at least `0.48`, and
  Theil-Sen slope no worse than `-0.005` score per 100 updates. Keep p4870 as
  the selected parent unless an external panel establishes a real upgrade.
- **Reject:** integrity or throughput fails; direct endpoint/final-three mean is
  below `0.47`; paired p4870 or heldout 80% lower bound is worse than -3 points;
  heldout Water falls over 5 points; Water proactive/response use falls over
  3/5 points; portal rate falls below 90%; attack rate falls over 4 points; mean
  game length rises over 15%; unique cards fall by more than two; or four-of
  share rises over five points.

Garden remains diagnostic rather than an adoption gate because the exact
Garden counterfactual was neutral-negative. A passing endpoint should be
followed by a separate matched extension at zero before making zero shaping the
default; it should not be combined with the rejected Step 5b estimator.

## Execution Log

The first full attempt, `shaped_reward_latezero20_v1`, stopped at the
pre-treatment fork gate. Its shared p6900 trunk scored `0.296875` over 192
paired games against p4870 with an 80% paired lower bound of `0.270833` and no
timeouts. No branch trained, so this result says nothing about late annealing.

The failure audit found the strength loss emerging between p6000 and p6500,
with no corresponding KL, clipping, entropy, value-loss, action-mix, LR, SPS,
or parameter-norm anomaly. Leader and whole-draft terminal-credit examples,
losses, gradients, and GPU time were all exactly zero. The disabled Step 5b
path returns before draft decoding, capture, label, or update work and does not
consume RNG. Two earlier independently executed p4870-p6000 floor runs also
diverged by essentially the same network parameter distance despite identical
seeds and rewards, while both retained roughly 44% parent-panel scores. This is
consistent with stochastic self-play lineage variance rather than a hidden
reward change.

Run one clean retry as `shaped_reward_latezero20_retry1_v1` at the registered
`0.00003` peak. Reusing the same LR avoids changing treatment sensitivity after
seeing a pre-treatment result. Apply the same fork qualification without
relaxation; a second failure triggers a separate continuation-stability design
rather than repeated selection for a fortunate trunk.

## Completed Result

The clean retry passed its unchanged p6900 qualification gate at `48.44%`
against p4870 with an 80% paired lower bound of `46.35%` over 192 games. The
forked model and trainer state were byte-identical. Both arms then reached
p7800 with the registered reward stack and cosine LR schedule.

The treatment reached an exact multiplier of zero at p7200 and remained at
zero through p7800. Raw telemetry contains 601 unique zero-multiplier updates,
or `20.51%` of the p4871-p7800 continuation. At p7199 the effective shaping
scale was `0.0005`; at p7200 it was exactly `0.0`. The p7200 process restart
restored that schedule, and all subsequent observed effective scales remained
zero. Native shaping stayed at `0.15`, terminal win labels remained active
with as many as 1,918 labeled rows in one zero-tail update, and leader and
whole-draft terminal credit remained exactly disabled.

Median steady SPS was `1452.83` for fixed-floor and `1446.96` for late-zero.
Treatment therefore retained `99.60%` of control throughput and cleared both
the registered relative and absolute performance gates.

### Terminal trajectory

| Epoch | Late-zero vs fixed-floor | LCB80 |
|---:|---:|---:|
| 7200 | 45.31% | 43.23% |
| 7300 | 52.08% | 50.52% |
| 7400 | 50.52% | 48.96% |
| 7500 | 50.52% | 48.96% |
| 7600 | 50.52% | 48.96% |
| 7700 | 49.48% | 47.40% |
| 7800 | 50.00% | 47.92% |

The final-three mean was `50.00%`. The robust Theil-Sen slope was `-0.13`
points per 100 updates while OLS was `+0.32` points per 100 updates. This is a
recovery from the initial ramp-end dip followed by oscillation around parity,
not evidence of a sustained upward terminal-strength trend.

At p7800, late-zero scored `47.40%` against p4870 versus fixed-floor's
`44.79%`, a paired delta of `+2.60` points with an 80% interval from `-1.04`
to `+6.25`. Heldout-reference score was `67.71%` versus `64.58%`, a paired
delta of `+3.13` points with an 80% interval from `+0.69` to `+5.56`; the 95%
interval still crossed zero. Training-reference score moved by only `-0.69`
points. These external panels are favorable, but they do not override the
direct parity result or establish a new accepted parent.

### Decks and mechanics

All registered mechanics gates passed. Relative to fixed-floor at p7800,
late-zero increased attack rate by `0.65` points, retained `105.5%` of portal
rate, reduced mean game length by `2.0%`, and left deterministic unique-card
count and four-of share unchanged. Heldout Water moved by `-1.39` points.
Opportunity-normalized Water proactive and response spell selection improved
by `2.43` and `8.33` points respectively.

The greedy late-zero Water deck contained no spell while fixed-floor used four
copies of Pulled Under, so that deterministic deck remains worth monitoring.
It was not a sampled-policy or battle-mechanics collapse: sampled Water spell
slots moved only from `18.6%` to `17.5%`, and both conditional spell-use rates
improved. Direct Garden selection remained rare in both arms and is still only
diagnostic, consistent with the neutral-negative Garden counterfactual.

## Decision And Retention

The registered decision is **zero viable but neutral**. This later schedule
avoided the material strength and mechanics losses from the rejected broad
30.7% zero-tail run, and it proves that a mature policy can complete roughly
the final fifth of training with no shaped reward without collapsing. It does
not yet show that terminal-only training keeps improving, finds a superior
optimum, or should replace the `0.15` floor by default.

Keep p4870 as the selected atomic parent. Archive the late-zero p7800 model as
a continuation candidate, not a promotion. Before making zero shaping the
default, run the separately preregistered short matched stability extension at
zero; do not combine it with the rejected Step 5b estimator. Full reports are
in `results/shaped_reward_latezero20_retry1_v1/trajectory_report.md` and
`results/shaped_reward_latezero20_retry1_v1/deck_compositions.md`.
