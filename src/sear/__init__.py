"""SEAR: Safe Evolution of Agentic Robots.

Package layout follows the proposal's work packages:

* :mod:`sear.core`         -- evolving system state z_k, update operator U, versioning
* :mod:`sear.contracts`    -- WP1, immutable machine-checkable safety contracts
* :mod:`sear.verification` -- WP2, evolution-aware candidate verification
* :mod:`sear.runtime`      -- WP3, runtime assurance and safe fallback
* :mod:`sear.lineage`      -- WP4, evolution memory, evidence lineage, rollback
* :mod:`sear.evolution`    -- the agentic improvement loop and baselines
"""

__version__ = "0.1.0"
