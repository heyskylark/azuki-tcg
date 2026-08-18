# Regional Reference-Seat 15M Readout

SPS is reported but is not a rejection criterion. Scores use identical paired schedules.

| Arm | Sampled rows | SPS median | Fixed-deck games | Train refs | Holdout refs | H2H parent |
|---|---:|---:|---:|---:|---:|---:|
| control | 14,899,200 | 1330 | 0 | 0.590 | 0.610 | 0.510 |
| ref05 | 14,899,200 | 1325 | 214 | 0.585 | 0.596 | 0.505 |

## Paired ref05 minus control

- **train_reference:** `-0.0056`; 80% CI `-0.0127` to `+0.0015`; 95% CI `-0.0165` to `+0.0052`.
- **holdout_reference:** `-0.0146`; 80% CI `-0.0437` to `+0.0146`; 95% CI `-0.0583` to `+0.0312`.
- **h2h_vs_parent:** `-0.0052`; 80% CI `-0.0469` to `+0.0365`; 95% CI `-0.0729` to `+0.0625`.

## Decision

**stop_after_15m**

Continue only when reference-seat integrity and zero-timeout checks pass, deck diversity does not collapse, at least two of train-reference, signature-heldout, and H2H improve by 2 points, the heldout 80% lower bound is at least -2 points, and H2H is at least 50%.
