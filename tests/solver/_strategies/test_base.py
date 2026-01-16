import numpy as np

from max_div.solver._strategies._base import StrategyBase


def test_strategy_base_seed_default():
    """Test if default seed is deterministic but depending on name"""

    # --- arrange -----------------------------------------
    strategy_1 = StrategyBase(name="s1")
    strategy_2a = StrategyBase(name="s2")
    strategy_2b = StrategyBase(name="s2")

    strategy_3a = StrategyBase()
    strategy_3b = StrategyBase()

    # --- act ---------------------------------------------
    seed_1 = strategy_1.seed
    seed_2a = strategy_2a.seed
    seed_2b = strategy_2b.seed
    seed_3a = strategy_3a.seed
    seed_3b = strategy_3b.seed

    # --- assert ------------------------------------------
    assert seed_1 != seed_2a
    assert seed_1 != seed_2b
    assert seed_1 != seed_3a
    assert seed_1 != seed_3b
    assert seed_2a != seed_3a
    assert seed_2a == seed_2b
    assert seed_3a == seed_3b


def test_strategy_base_seed_default_unique():
    """100 different names should lead to 100 different seeds"""
    assert len({StrategyBase(name=str(i)).seed for i in range(100)}) == 100


def test_strategy_base_set_seed():
    """Test if set_seed is deterministic"""

    # --- arrange -----------------------------------------
    strategy = StrategyBase()

    # --- act ---------------------------------------------
    seed_1 = strategy.seed
    rng_state_1 = strategy._rng_state.copy()
    strategy.set_seed(42)
    seed_2 = strategy.seed
    rng_state_2 = strategy._rng_state.copy()
    strategy.set_seed(43)
    seed_3 = strategy.seed
    seed_3b = strategy.seed  # should stay the same
    rng_state_3 = strategy._rng_state.copy()

    # --- assert ------------------------------------------
    assert seed_1 != seed_2
    assert seed_2 != seed_3
    assert seed_2 == 42
    assert seed_3 == 43
    assert seed_3b == 43
    assert not np.array_equal(rng_state_1, rng_state_2)
    assert not np.array_equal(rng_state_2, rng_state_3)
