"""Agentic self-improvement loop, proposers and baselines."""

from sear.evolution.evaluator import DEFAULT_EVALUATOR, TaskEvaluator
from sear.evolution.experiment import METHODS, build_loop, load_config, make_genesis, run_experiment
from sear.evolution.loop import EvolutionConfig, EvolutionLoop, EvolutionResult, GenerationRecord
from sear.evolution.proposer import (
    ATTACKS,
    AdversarialProposer,
    ExternalAgentProposer,
    Feedback,
    GreedyTuningProposer,
    Proposer,
)

__all__ = [
    "ATTACKS",
    "DEFAULT_EVALUATOR",
    "METHODS",
    "AdversarialProposer",
    "EvolutionConfig",
    "EvolutionLoop",
    "EvolutionResult",
    "ExternalAgentProposer",
    "Feedback",
    "GenerationRecord",
    "GreedyTuningProposer",
    "Proposer",
    "TaskEvaluator",
    "build_loop",
    "load_config",
    "make_genesis",
    "run_experiment",
]
