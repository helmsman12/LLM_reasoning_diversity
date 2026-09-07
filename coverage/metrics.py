"""Approach coverage metrics.

Pure module: no I/O, no model calls. All functions operate on a single
problem's data (problem-local approach labels) and return scalars or dicts.

See ``CLAUDE.md`` for the formal definitions and conventions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

NAN = float("nan")
DEFAULT_N_VALUES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


def approach_frequencies(labels: Sequence[int]) -> List[int]:
    """Return the frequency vector of approach labels.

    Order is by first appearance of each label, which keeps results
    deterministic but is not relied upon by downstream metrics.
    """
    counts: "Counter[int]" = Counter()
    order: List[int] = []
    for lab in labels:
        if lab not in counts:
            order.append(lab)
        counts[lab] += 1
    return [counts[lab] for lab in order]


def coverage_at_n(freqs: Sequence[int], N: int) -> float:
    """Analytic unbiased estimator of E[# distinct approaches in N draws].

    Sampling is **without replacement** from the empirical pool of correct
    solutions (size ``n_correct = sum(freqs)``). For each approach ``k``
    with frequency ``f_k``, the probability that it is *missed* in a draw
    of size N is ``C(n_correct - f_k, N) / C(n_correct, N)``. Summing the
    complement over all approaches gives the expected number of distinct
    approaches observed.

    Returns ``NaN`` if ``N > n_correct`` or ``n_correct == 0`` (NaN
    discipline — never impute, interpolate, or substitute zero).
    """
    n_correct = sum(freqs)
    if n_correct == 0:
        return NAN
    if N > n_correct:
        return NAN
    if N <= 0:
        return 0.0

    total = math.comb(n_correct, N)
    expected = 0.0
    for f_k in freqs:
        # Probability approach k is absent from a size-N sample.
        miss = math.comb(n_correct - f_k, N) / total
        expected += 1.0 - miss
    return expected


def coverage_curve(
    labels: Sequence[int],
    n_values: Iterable[int] = DEFAULT_N_VALUES,
) -> Dict[int, float]:
    """Coverage@N evaluated at each N in ``n_values``.

    NaN is returned for N > n_correct (NaN discipline).
    """
    freqs = approach_frequencies(labels)
    return {int(N): coverage_at_n(freqs, int(N)) for N in n_values}


def compute_all(
    labels: Sequence[int],
    n_values: Iterable[int] = DEFAULT_N_VALUES,
) -> Dict[str, object]:
    """Compute the ``metrics`` field for one problem record.

    ``labels`` is the list of problem-local approach IDs over the
    correct solutions only. When ``n_correct == 0`` every Coverage@N is
    NaN, matching the ``no_correct_solutions`` status convention.
    """
    return {"cov_at_n": coverage_curve(labels, n_values=n_values)}
