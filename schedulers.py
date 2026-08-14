"""Scheduler construction shared by run_simulations.py and
run_combined_simulations.py. Kept in its own module (rather than defined in
run_simulations.py) because that script has module-level argument parsing
that runs on import -- importing from it directly would re-parse argv."""
from __future__ import annotations

from syne_tune.optimizer.baselines import ASHA, BOHB, BOTorch, CQR, RandomSearch, TPE

from common import FIDELITY_ATTR


def make_scheduler(name: str, config_space: dict, metric: str, max_t: int, min_t: int, seed: int):
    if name == "RandomSearch":
        return RandomSearch(config_space=config_space, metrics=[metric], do_minimize=False, random_seed=seed)
    if name == "BOTorch":
        return BOTorch(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "TPE":
        return TPE(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "CQR":
        return CQR(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "ASHA":
        return ASHA(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=min_t, reduction_factor=2, do_minimize=False, random_seed=seed,
        )
    if name == "BOHB":
        return BOHB(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=min_t, reduction_factor=2, do_minimize=False, random_seed=seed,
        )
    raise ValueError(name)
