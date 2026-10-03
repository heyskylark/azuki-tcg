from pathlib import Path

import pytest

from training_utils import load_training_config


def test_training_rejects_a_different_potential_discount(tmp_path: Path) -> None:
    config = tmp_path / "mismatched.ini"
    config.write_text(
        "[env]\nnative = true\npbrs_mode = discounted\npbrs_gamma = 0.99\n"
        "[train]\ngamma = 1.0\n"
    )
    with pytest.raises(ValueError, match="must match train.gamma"):
        load_training_config(config, [])


def test_training_rejects_unclosed_discounted_potential(tmp_path: Path) -> None:
    config = tmp_path / "unclosed.ini"
    config.write_text(
        "[env]\nnative = true\npbrs_mode = discounted\n"
        "pbrs_terminal_closure = false\n[train]\ngamma = 0.99\n"
    )
    with pytest.raises(ValueError, match="requires terminal closure"):
        load_training_config(config, [])
