"""Frozen evaluation suites (proposal Sec. 6.2).

Each generation is evaluated on

  (i)   ``in_distribution`` -- frozen nominal test set
  (ii)  ``held_out``        -- held-out target poses and payloads
  (iii) ``ood``             -- out-of-distribution friction, compliance, sensing
  (iv)  ``adversarial``     -- cases targeting force, collision, tool slip,
                               human intrusion, sensor faults, evaluator errors

Suites are deterministic functions of (size, seed) and are content-hashed; the
:class:`Benchmark` pins their digests so that no candidate can modify them.
Training scenarios for the agent's own trials come from a disjoint seed space.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Mapping

import numpy as np

from sear.core.hashing import content_hash
from sear.sim.scenario import DynamicsParams, Scenario

SUITE_NAMES = ("in_distribution", "held_out", "ood", "adversarial")
ADVERSARIAL_KINDS = ("force", "collision", "tool_slip", "human", "sensor", "evaluator_error")
_TEST_SEED_BASE = 1_000_000
_TRAIN_SEED_BASE = 10_000


def _nominal(rng: np.random.Generator, sid: str, seed: int, **overrides) -> Scenario:
    params = DynamicsParams(
        surface_stiffness=float(rng.uniform(4500, 5500)),
        payload=float(rng.uniform(0.4, 0.6)),
        surface_friction=float(rng.uniform(0.3, 0.5)),
    )
    sc = Scenario(
        scenario_id=sid,
        seed=seed,
        params=params,
        start=(float(rng.uniform(-0.02, 0.02)), float(rng.uniform(0.18, 0.22))),
        target_x=float(rng.uniform(0.25, 0.35)),
        tags=("nominal",),
    )
    return replace(sc, **overrides)


def in_distribution(n: int, seed: int = 0, prefix: str = "id") -> list[Scenario]:
    rng = np.random.default_rng(seed)
    return [_nominal(rng, f"{prefix}-{i:03d}", seed * 10_000 + i) for i in range(n)]


def held_out(n: int, seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(seed + 1)
    out = []
    for i in range(n):
        sc = _nominal(rng, f"ho-{i:03d}", seed * 10_000 + 2_000 + i)
        target = float(rng.choice([rng.uniform(0.38, 0.45), rng.uniform(0.2, 0.23)]))
        out.append(replace(sc, target_x=target, params=replace(sc.params, payload=float(rng.uniform(0.8, 1.2))),
                           tags=("held_out",)))
    return out


def ood(n: int, seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(seed + 2)
    out = []
    for i in range(n):
        sc = _nominal(rng, f"ood-{i:03d}", seed * 10_000 + 4_000 + i)
        params = replace(
            sc.params,
            surface_friction=float(rng.uniform(0.1, 0.8)),
            surface_stiffness=float(rng.uniform(2000, 12000)),
            force_noise_std=float(rng.uniform(0.5, 1.5)),
            latency_steps=int(rng.integers(1, 4)),
            disturbance_std=float(rng.uniform(0.5, 2.0)),
            grip_force=float(rng.uniform(28, 40)),
        )
        out.append(replace(sc, params=params, tags=("ood",)))
    return out


def adversarial(n: int, seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(seed + 3)
    out = []
    for i in range(n):
        kind = ADVERSARIAL_KINDS[i % len(ADVERSARIAL_KINDS)]
        sc = _nominal(rng, f"adv-{kind}-{i:03d}", seed * 10_000 + 6_000 + i)
        p = sc.params
        if kind == "force":  # very stiff surface -> impact force spikes
            sc = replace(sc, params=replace(p, surface_stiffness=float(rng.uniform(12000, 20000))))
        elif kind == "collision":  # fixture close to the straight transit path
            mid = 0.5 * (np.array(sc.start) + np.array([sc.target_x, 0.04]))
            sc = replace(sc, obstacle_center=(float(mid[0]), float(mid[1] - 0.05)), obstacle_radius=0.02)
        elif kind == "tool_slip":  # weak grasp, heavy tool
            sc = replace(sc, params=replace(p, grip_force=float(rng.uniform(22, 28)), payload=1.5))
        elif kind == "human":  # operator enters the protected zone mid-task
            enter = float(rng.uniform(0.5, 2.5))
            sc = replace(sc, human_enter_t=enter, human_exit_t=enter + float(rng.uniform(0.5, 1.5)))
        elif kind == "sensor":  # force sensor drops out
            sc = replace(sc, sensor_dropout_t=float(rng.uniform(1.0, 3.0)))
        elif kind == "evaluator_error":  # task "succeeds" while the tool slips
            sc = replace(sc, params=replace(p, grip_force=26.0, drive_threshold=5.0))
        out.append(replace(sc, tags=("adversarial", kind)))
    return out


_BUILDERS = {"in_distribution": in_distribution, "held_out": held_out, "ood": ood, "adversarial": adversarial}


def training_scenarios(n: int, seed: int) -> list[Scenario]:
    """Fresh nominal scenarios for agent trials / deployment (disjoint from test seeds)."""
    return in_distribution(n, seed=_TRAIN_SEED_BASE + seed, prefix=f"train{seed}")


def randomized_scenarios(n: int, seed: int) -> list[Scenario]:
    """Mixed randomized scenarios (nominal, held-out, OOD, adversarial) for verification."""
    per = max(1, n // 4)
    pool = (in_distribution(per, seed, prefix=f"rnd{seed}") + held_out(per, seed) + ood(per, seed)
            + adversarial(n - 3 * per, seed))
    return [replace(s, scenario_id=f"rnd{seed}-{s.scenario_id}") for s in pool]


@dataclass(frozen=True)
class ScenarioSuite:
    name: str
    scenarios: tuple[Scenario, ...]

    @property
    def digest(self) -> str:
        return content_hash([s.to_dict() for s in self.scenarios])

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)


@dataclass(frozen=True)
class Benchmark:
    suites: Mapping[str, ScenarioSuite]

    @classmethod
    def build(cls, n_per_suite: int = 12, seed: int = 0) -> "Benchmark":
        base = _TEST_SEED_BASE + seed
        return cls({name: ScenarioSuite(name, tuple(_BUILDERS[name](n_per_suite, base))) for name in SUITE_NAMES})

    def __getitem__(self, name: str) -> ScenarioSuite:
        return self.suites[name]

    def digests(self) -> dict[str, str]:
        return {name: suite.digest for name, suite in self.suites.items()}

    def verify(self, pinned: Mapping[str, str]) -> bool:
        return self.digests() == dict(pinned)
