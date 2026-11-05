"""Task and safety metrics with honest uncertainty quantification."""

from sear.metrics.episode import EpisodeOutcome, SuiteMetrics, aggregate, summarize_episode
from sear.metrics.stats import (
    clopper_pearson,
    clopper_pearson_lower,
    clopper_pearson_upper,
    mean_ci,
    poisson_rate_upper,
    wilson_interval,
)

__all__ = [
    "EpisodeOutcome",
    "SuiteMetrics",
    "aggregate",
    "clopper_pearson",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "mean_ci",
    "poisson_rate_upper",
    "summarize_episode",
    "wilson_interval",
]
