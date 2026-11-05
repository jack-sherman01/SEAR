from sear.benchmark import ADVERSARIAL_KINDS, SUITE_NAMES, Benchmark, training_scenarios
from sear.sim import Scenario


def test_benchmark_is_deterministic_and_pinned():
    a, b = Benchmark.build(6, seed=0), Benchmark.build(6, seed=0)
    assert a.digests() == b.digests()
    assert a.verify(b.digests())
    assert not Benchmark.build(6, seed=1).verify(a.digests())
    assert set(a.suites) == set(SUITE_NAMES)


def test_adversarial_covers_all_kinds():
    bench = Benchmark.build(len(ADVERSARIAL_KINDS))
    kinds = {s.tags[1] for s in bench["adversarial"]}
    assert kinds == set(ADVERSARIAL_KINDS)


def test_training_seeds_disjoint_from_test():
    bench = Benchmark.build(12)
    test_seeds = {s.seed for suite in bench.suites.values() for s in suite}
    train_seeds = {s.seed for g in range(20) for s in training_scenarios(8, g)}
    assert not test_seeds & train_seeds


def test_scenario_roundtrip():
    sc = Benchmark.build(3)["adversarial"].scenarios[0]
    assert Scenario.from_dict(sc.to_dict()) == sc
