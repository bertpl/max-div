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


def test_strategy_base_seed_set():
    """Test if set seed is deterministic"""

    # --- arrange -----------------------------------------
    strategy = StrategyBase()

    # --- act ---------------------------------------------
    seed_1 = strategy.seed
    strategy.seed = 42
    seed_2 = strategy.seed
    strategy.seed = 43
    seed_3 = strategy.seed
    seed_3b = strategy.seed  # should stay the same

    # --- assert ------------------------------------------
    assert seed_1 != seed_2
    assert seed_2 != seed_3
    assert seed_2 == 42
    assert seed_3 == 43
    assert seed_3b == 43


def test_strategy_base_seed_auto_update():
    """Test if the seed auto-updates upon each access"""

    # --- arrange -----------------------------------------
    strategy = StrategyBase()

    # --- act ---------------------------------------------
    seeds = [strategy.next_seed() for _ in range(100)]

    # --- assert ------------------------------------------
    assert len(seeds) == len(set(seeds)), "100 accessed seeds should all be unique"
