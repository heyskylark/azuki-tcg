"""S4 reference-seat (AZK_DRAFT_REF_SEAT_PROB) differential tests.

prob=1 with AZK_DRAFT_REF_DECK_INDICES pinned to one deck: every episode must
carry a ref seat whose exported gate/leader/mains reproduce that pool spec
exactly (IKZ excluded), the drafter seat must still complete a full 50-card
draft, and episodes must terminate (a ref seat that were ever asked to pick
would abort the env). prob unset: every record reports ref_seat == -1.

Run: PYTHONPATH=build/python/src:python/src pytest \
    python/src/test_draft_ref_seat.py -q
"""

from __future__ import annotations

import numpy as np

REF_DECK_INDEX = 3


def _worker(env_updates: dict, min_records: int, queue) -> None:
    try:
        import os

        for key in ("AZK_DRAFT_REF_SEAT_PROB", "AZK_DRAFT_REF_DECK_INDICES",
                    "AZK_DEBUG_FORCE_GATE_DEF_IDS", "AZK_DECKBUILD_SNAPSHOT_DIR",
                    "AZK_PORTAL_GP_BONUS", "AZK_EARLY_TEMPO_BONUS",
                    "AZK_DMG_MITIGATION_BONUS"):
            os.environ.pop(key, None)
        os.environ.update(env_updates)
        import binding
        from azk_native import AzukiNativeEnv, NATIVE_DECKBUILD_OBS_DTYPE
        from deck_building import build_deck_build_catalog
        from training_deck_pool import load_training_deck_pool

        pool = load_training_deck_pool()
        env = AzukiNativeEnv(num_envs=1, deck_pool=pool, deck_building=True, seed=13)
        env.reset(seed=13)
        # The wrapper drains records into its metrics helper on terminal steps;
        # disable it so this test sees the raw per-episode records.
        env._deckbuild_helper = None
        view = env.observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(env.num_agents)
        rng = np.random.default_rng(77)
        records = []
        for _ in range(60000):
            acts = np.zeros((env.num_agents, 4), dtype=np.int32)
            for row in range(env.num_agents):
                am = view[row]["action_mask"]
                count = int(am["legal_action_count"])
                if count > 0:
                    r = int(rng.integers(0, count))
                    acts[row] = (
                        am["legal_primary"][r], am["legal_sub1"][r],
                        am["legal_sub2"][r], am["legal_sub3"][r],
                    )
            env.actions[:] = acts.reshape(env.actions.shape)
            env.step()
            records.extend(binding.vec_drain_deck_records(env._handle))
            if len(records) >= min_records:
                break
        env.close()

        catalog = build_deck_build_catalog(pool)
        code_to_def = {r.card_code: d for d, r in catalog.records_by_def_id.items()}

        def spec_expected(deck):
            leader = gate = None
            mains: list[int] = []
            for code, qty in deck:
                def_id = code_to_def.get(code)
                rec = catalog.records_by_def_id.get(def_id) if def_id is not None else None
                card_type = rec.card_type if rec else "?"
                if card_type == "LEADER":
                    leader = def_id
                elif card_type == "GATE":
                    gate = def_id
                elif card_type == "IKZ":
                    continue
                else:
                    mains.extend([def_id] * int(qty))
            return leader, gate, sorted(mains)

        queue.put((records, spec_expected(pool[REF_DECK_INDEX])))
    except Exception:
        import traceback

        queue.put(traceback.format_exc())


def _run(env_updates: dict, min_records: int):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(env_updates, min_records, queue))
    proc.start()
    out = queue.get(timeout=900)
    proc.join(timeout=60)
    if isinstance(out, str):
        raise RuntimeError(out)
    return out


def test_ref_seat_pinned_deck():
    records, (exp_leader, exp_gate, exp_mains) = _run(
        {"AZK_DRAFT_REF_SEAT_PROB": "1", "AZK_DRAFT_REF_DECK_INDICES": str(REF_DECK_INDEX)},
        min_records=6,
    )
    assert len(records) >= 6
    assert exp_leader is not None and exp_gate is not None and len(exp_mains) == 50
    seats_seen = set()
    for rec in records:
        ref_seat = int(rec["ref_seat"])
        assert ref_seat in (0, 1), f"expected a ref seat every episode, got {ref_seat}"
        assert int(rec["ref_deck_index"]) == REF_DECK_INDEX
        seats_seen.add(ref_seat)
        players = rec["players"]
        ref = players[ref_seat]
        assert int(ref["leader"]) == exp_leader
        assert int(ref["gate"]) == exp_gate
        got_mains = sorted(int(c) for c in ref["main"] if int(c) >= 0)
        assert got_mains == exp_mains, "ref seat mains do not match the pool spec"
        drafter = players[1 - ref_seat]
        drafter_mains = [int(c) for c in drafter["main"] if int(c) >= 0]
        assert len(drafter_mains) == 50, "drafter seat did not complete its draft"
    assert seats_seen == {0, 1}, f"ref seat never alternated: {seats_seen}"


def test_ref_seat_off_by_default():
    records, _ = _run({}, min_records=4)
    assert len(records) >= 4
    for rec in records:
        assert int(rec["ref_seat"]) == -1
        assert int(rec["ref_deck_index"]) == -1
