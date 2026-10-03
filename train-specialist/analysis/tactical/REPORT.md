# Tactical fixture benchmark: element specialists vs u8223 / u9305

These are constructed, legal-history conditional tactics. They are not natural-game frequencies and they are not win rates. Every policy replays the same complete legal prefix through its own recurrent state. It then acts until the case horizon: greedy (argmax) once, plus multinomial samples (4 seeds for the Lightning port, matching the original; 16 seeds `91001–91016` for the new families). Grading uses the real engine outcome: a win, survival through the opponent's turn, a favorable trade, or restraint. Activation counts are recorded but are not the grade.

## 1. Lightning port (`lightning/`, the original 56 draft-history fixtures)
- **No drift.** `drift_check.py` replays all 56 frozen v3 prefixes in the snapshot runtime: 56/56 stay legal and all checkpoint snapshots match byte for byte, including the legal-action lists. Regenerating with the original selection rule gives the identical `cases.json`. The oracle passes 140/140 branches with outcomes identical to v3.
- The ported u8223 and u9305 runs reproduce the original `seed43_step20m` and `seed43_step30m` rollouts exactly: 328/328 identical continuations, the same greedy root action, and max |ΔP(leader)| below 1e-29. The history hashes differ at the byte level because the observation encoding changed.
- Results: across all 6 policies (u8223, u9305, Lightning, plus Fire, Water and Earth as off-element controls), every policy scores 0 on ability-dependent lethal (0/8 greedy, 0/32 sampled), setup-dependent lethal (0/8, 0/32) and favorable trade (0/16, 0/64). Every policy scores 100% on simpler wins and on the pointless-Piko cases. **There are 0 leader activations in any rollout.** Paired Δ = 0.

## 2. New families (curated constructed decks, 2 seats × ≥2 world seeds; each case is oracle-verified)
Each cell shows greedy k/n · sampled k/n. L = leader activations in the sampled rollouts; H = harmful activations or self-kills. Δ is the specialist minus u8223 on sampled rollouts, with a cluster-bootstrap 95% CI over fixture instances.

| family / case (why it is hard) | u8223 | u9305 | specialist | off-element | Δ [CI] |
|---|---|---|---|---|---|
| **Lightning, constructed** raizan_charge_lethal (needs Raizan Charge) | 0/6 · 0/96 | 0/6 · 0/96 | 1/6 · 16/96 (L16) | Fire 0/96 | +0.17 [0, +0.50] |
| surge_lethal (portal, then a weapon from the discard) | 6/6 · 96/96 | 96/96 | 96/96 | 96/96 | 0 |
| **Fire** zero_lethal (ping, then attack) | 0/4 · 0/64 (L63) | 0/64 (L62) | 0/64 (L14) | Water 0/64 (L64) | 0 |
| zero_ritualist_burn (Zero, then the Ritualist trigger hits face) | 0/64 (L48) | 0/64 (L32) | 0/64 (L0) | 0/64 (L64) | 0 |
| ragefire_combo (Zero + Ragefire) | 0/64 (L64) | 0/64 | 0/64 (L14) | 0/64 | 0 |
| rushfire_lethal (Charge payload) | 4/4 · 64/64 | 64/64 | 3/4 · 51/64 | 64/64 | −0.20 [−0.61, 0] |
| zero_suicide restraint (1 HP, no self-damage) | 1/4 · 15/64 (H49) | 28/64 (H36) | 2/4 · **43/64 (H21)** | 0/64 (H64) | **+0.44 [+0.03, +0.84]** |
| zero_equivalent (control) | 64/64 | 64/64 | 64/64 | 64/64 | 0 |
| **Water** hydro_tenshin_lethal (portal refunds 2 IKZ, then Tenshin) | 4/4 · 64/64 | 64/64 | 64/64 | 64/64 | 0 |
| hold_ikz (keep the last IKZ for Shao) | 0/4 · 0/64 | 0/64 | 0/64 | 0/64 | 0 |
| shao_response (respond to lethal) | 64/64 (L52) | 64/64 (L48) | 64/64 (L64) | 64/64 | 0 |
| hydro_equivalent (control) | 64/64 | 64/64 | 64/64 | 64/64 | 0 |
| **Earth** stonehaven_block (portal, grant Defender, block) | 4/4 · 64/64 | 64/64 | 3/4 · 49/64 | Lightning 64/64 | −0.23 [−0.70, 0] |
| declare_defender (block in the response window) | 64/64 | 64/64 | 64/64 | 64/64 | 0 |
| quicksand_lethal (clear the Foamback blocker) | 0/64 (H5) | 0/64 | 0/64 (H3) | 0/64 | 0 |
| goro_trade (+1 HP turns an even trade favorable) | 0/64 (L0) | 0/64 | 0/64 (L1) | 0/64 | 0 |
| stonehaven_unneeded (control) | 64/64 | 64/64 | 64/64 | 64/64 | 0 |

Pooled over all of a family's cases, Δ (specialist − u8223), greedy · sampled:

| family | greedy Δ [95% CI] | sampled Δ [95% CI] |
|---|---|---|
| Lightning port | 0 | 0 |
| Lightning, constructed | +0.08 [0, +0.25] | +0.08 [0, +0.25] |
| Fire | 0.00 [−0.12, +0.12] | +0.04 [−0.07, +0.16] |
| Water | 0 | 0 |
| Earth | −0.05 [−0.15, 0] | −0.05 [−0.14, 0] |

The comparisons against u9305 give the same picture. Full tables, including clopper-pearson intervals, are in `tables.md`, `compact_tables.md` and `summary.json`.

## 3. Did each specialist learn the conditional tactics u8223 lacks?
- **Lightning:** only partly, and only on constructed-deck histories.
  - On one of 6 Raizan instances (seat 1, seed 644) the specialist's root P(Raizan) is 0.998, against 0.0008 for u8223. It wins there in 17/17 rollouts with: `LEADER_ABILITY > TARGET(Prowler) > ATTACK(Prowler→FACE)`.
  - u8223 on the same board plays `PLAY→alley(Recruit) > PASS`.
  - On the other 5 instances, and on all draft-history fixtures, it behaves like u8223.
- **Fire:** no. It learned restraint, not the combos.
  - It cut Zero activations sharply: on zero_ritualist_burn the root P(Zero) falls from 0.74 to 0.0005, and misdirected or post-attack activations on zero_lethal fall from 63/64 to 14/64. It also self-kills less often at 1 HP (21/64 vs 49/64).
  - It solves none of the Zero-dependent lines. u8223 activates Zero but in the wrong order: `ATTACK(Rei→FACE) > LEADER_ABILITY > TARGET(alley Pekiro)`. The Fire specialist simply attacks and passes.
  - New error on rushfire_lethal: the specialist portals the gate-power-0 Cinderwake Seer, so no payload is possible: `PORTAL(Seer) > PASS`.
- **Water:** no new tactic.
  - Shao response is sharper (root P 0.83 → 1.00, used in 64/64 vs 52/64 rollouts), but every policy already survives that case.
  - No policy plans ahead to hold the IKZ: in 100% of rollouts the first action spends it (Tidal Insight or a 1-drop), then the policy dies.
- **Earth:** no, and slightly worse.
  - Goro is never used to save the attacker, and nobody casts Quicksand: all policies attack into the Foamback block.
  - The Stonehaven block regresses at one instance. The specialist grants Defender, then passes when the `[9,0]` block row is legal at every attack.

## 4. Caveats
- n is small: 4–6 fixture instances per case type. Sampled repetitions share fixtures, so the CIs come from a fixture-cluster bootstrap. The clopper-pearson intervals in `tables.md` overstate precision.
- The opponent is scripted: passive in our turn, aggressive (attack everything) in its own. It blocks only in quicksand_lethal.
- The oracles prove that the intended line works and that the listed alternatives fail. They are not exhaustive.
- Three fixture flaws were found by off-element anomalies and fixed, then the affected families were regenerated. The old versions are kept in `superseded/`:
  - native-Defender bodies placed on the board during setup;
  - a Tumbleweed sacrifice-removal alternative in goro_trade;
  - a Black Jade Dagger leader-lethal alternative in raizan_charge_lethal.
- Dropped: a Bobu heal case (the effect only heals 1 when an Earth entity dies; no decisive fixture) and a Water bounce-lethal case (it needs a blocking opponent with Penny; not built).
- Natural-game strategy is not measured here.

## 5. Commands (cwd = this dir; `LD_LIBRARY_PATH=<rt>/build/_deps/flecs_src-build`, python `<rt>/.venv/bin/python`; tb_runtime sets sys.path; CPU inference)
```
python lightning/drift_check.py                       # v3 fixture replay drift
python lightning/history_benchmark.py --generate      # regenerate (identical to v3)
python lightning/evaluate_history.py --oracle
python lightning/evaluate_history.py --policies --model {u8223,u9305,lightning,fire,water,earth}   # 4 samples, as original
python generate.py {fire|water|earth} --seed-stop 900                       # first 2 seeds/seat that build + verify
python generate.py lightning_c --seed-stop 1500 --per-seat 3
python tactical.py {fire|water|earth|lightning_c} --model <policy>          # greedy + 16 samples
python aggregate.py ; python traces.py <family> <case_id> <policy> [argmax|sample] [seed]
```

## 6. Files
- Framework: `tb_runtime.py` (snapshot-runtime runner/logger), `tb_fixture.py`, `tb_setup.py`, `tactical.py` (Brain, rollouts, oracle), `family_{fire,water,earth,lightning_c}.py` (case builders, oracle plans, documented why), `generate.py`, `aggregate.py`, `traces.py`, `card_stats.json` (engine card stats + registry text).
- Per family dir (`lightning/`, `lightning_c/`, `fire/`, `water/`, `earth/`): `cases.json`, `oracle_results.json`, `fixture_generation.json`, `<policy>_results.json`; `lightning/drift_v3_replay.json`, `lightning/original/` (frozen v3 cases/oracle).
- `summary.json`, `tables.md`, `compact_tables.md`, `superseded/` (pre-fix fixtures + results), `logs/`.
