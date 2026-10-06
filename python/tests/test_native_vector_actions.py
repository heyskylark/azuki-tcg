import os
import subprocess
import sys
import textwrap


def test_serial_native_vector_applies_supplied_draft_actions() -> None:
    # A missing shared-action-buffer write aborts in C. Keep the reproduction
    # in a subprocess so that regression cannot terminate the test runner.
    probe = textwrap.dedent(
        """
        import numpy as np
        from azk_native import NATIVE_DECKBUILD_OBS_DTYPE
        from training_utils import build_vecenv

        vec = build_vecenv({
            'env': {
                'native': True, 'native_envs_per_instance': 1,
                'deck_building_enabled': True, 'draft_uniform_assignment': True,
                'pbrs_mode': 'discounted',
            },
            'train': {'gamma': 0.99},
            'vec': {'backend': 'Serial', 'num_envs': 2},
        })
        try:
            vec.async_reset(seed=43)
            for step in range(5):
                observations, _, _, _, _, ids, _ = vec.recv()
                rows = observations.view(NATIVE_DECKBUILD_OBS_DTYPE).reshape(-1)
                assert int(rows['deck_context']['main_count'].sum()) == 2 * step
                if step == 4:
                    break
                actions = np.zeros((len(ids), 4), dtype=np.int32)
                for index, row in enumerate(rows):
                    mask = row['action_mask']
                    if mask['legal_action_count']:
                        actions[index] = [mask[key][0] for key in
                            ('legal_primary', 'legal_sub1', 'legal_sub2', 'legal_sub3')]
                vec.send(actions)
        finally:
            vec.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=os.environ.copy(), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
