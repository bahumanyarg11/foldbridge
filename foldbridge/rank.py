"""Rank scored models across backends/seeds/samples.

``composite`` is a weighted mean of per-metric z-scores (sign-flipped for lower-is-better
metrics) over the metrics present in the batch. Missing values contribute z = 0 (neutral).
With ``length_normalize`` and a batch of >= 8 rows whose sizes differ, each metric is first
regressed on log(total residues) across the batch and the residuals are z-scored, so that
size-correlated metrics (BSA, contact counts, pLDDT of small peptides, ...) do not just
reward bigger or smaller constructs.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["rank", "has_metrics", "pareto_front", "HIGHER_BETTER", "LOWER_BETTER", "DEFAULT_WEIGHTS"]

HIGHER_BETTER = ("iptm", "ptm", "plddt", "ipsae", "pdockq", "bsa", "dockq", "fnat", "ranking",
                 "plddt_interface", "n_contacts")
LOWER_BETTER = ("ipae", "clash_fraction", "pae_interchain", "rmsd", "irmsd", "lrmsd", "clash_count")

DEFAULT_WEIGHTS: dict[str, float] = {
    "iptm": 1.0,
    "ipsae": 1.0,
    "ptm": 0.5,
    "plddt": 0.5,
    "pdockq": 0.5,
    "ipae": 0.5,
    "bsa": 0.25,
    "clash_fraction": 0.5,
}

LENGTH_KEYS = ("n_residues", "n_tokens", "length")
MIN_BATCH_FOR_LENGTH_NORM = 8


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.array([_num(r.get(key)) for r in rows], dtype=np.float64)


def _lengths(rows: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    for k in LENGTH_KEYS:
        col = _column(rows, k)
        if np.isfinite(col).all() and (col > 0).all():
            return col
    return None


def _direction(metric: str) -> float:
    return -1.0 if metric in LOWER_BETTER else 1.0


def _zscore(x: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None
    mu, sd = x[ok].mean(), x[ok].std()
    if not sd > 1e-12:
        return None
    return np.where(ok, (x - mu) / sd, 0.0)


def _residualize(x: np.ndarray, loglen: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 3 or np.ptp(loglen[ok]) < 1e-12:
        return x
    A = np.stack([np.ones(ok.sum()), loglen[ok]], axis=1)
    coef, *_ = np.linalg.lstsq(A, x[ok], rcond=None)
    out = np.full_like(x, np.nan)
    out[ok] = x[ok] - A @ coef
    return out


def composite_scores(rows: Sequence[Mapping[str, Any]], length_normalize: bool = True,
                     weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Composite score per row (higher is better)."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    n = len(rows)
    total = np.zeros(n)
    wsum = 0.0
    loglen = None
    if length_normalize and n >= MIN_BATCH_FOR_LENGTH_NORM:
        L = _lengths(rows)
        if L is not None and np.ptp(L) > 0:
            loglen = np.log(L)
    for metric, wt in w.items():
        if wt == 0:
            continue
        x = _column(rows, metric)
        if loglen is not None:
            raw_sd = np.nanstd(x) if np.isfinite(x).any() else 0.0
            x = _residualize(x, loglen)
            # metric fully explained by size: residuals are round-off noise, drop it
            if np.isfinite(x).sum() >= 2 and np.nanstd(x) <= 1e-8 * raw_sd:
                continue
        z = _zscore(x)
        if z is None:
            continue
        total += wt * _direction(metric) * z
        wsum += abs(wt)
    return total / wsum if wsum > 0 else total


def has_metrics(row: Mapping[str, Any], keys: Iterable[str] | None = None) -> bool:
    """True if ``row`` holds at least one finite value for a rankable metric.

    Rows without any (e.g. a ``--dry-run`` result, or a failed model) cannot be ranked.
    """
    ks = set(keys) if keys is not None else set(HIGHER_BETTER) | set(LOWER_BETTER) | set(DEFAULT_WEIGHTS)
    return any(math.isfinite(_num(row.get(k))) for k in ks)


def rank(rows: Iterable[Mapping[str, Any]], by: str = "composite", length_normalize: bool = True,
         weights: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Return copies of ``rows`` with ``composite`` and ``rank`` (1 = best), sorted best first.

    ``by`` is "composite" or any metric key; known lower-is-better metrics (ipae, rmsd,
    clash_fraction, ...) sort ascending. Rows missing the sort key go last. ``weights``
    replaces :data:`DEFAULT_WEIGHTS` entirely when given (a negative weight flips a metric).

    Rows with no metric at all (dry-run results, failed models) are not ranked: they get
    ``rank = None`` and ``composite = None``, keep their input order, and come after every
    ranked row. They do not influence the z-scores of the other rows.
    """
    out = [dict(r) for r in rows]
    if not out:
        return out
    keys = set(HIGHER_BETTER) | set(LOWER_BETTER) | set(DEFAULT_WEIGHTS if weights is None else weights)
    if by != "composite":
        keys.add(by)
    scored = [r for r in out if has_metrics(r, keys)]
    unscored = [r for r in out if not has_metrics(r, keys)]
    comp = composite_scores(scored, length_normalize=length_normalize, weights=weights) if scored else []
    for r, c in zip(scored, comp):
        r["composite"] = float(c)
    for r in unscored:
        r["composite"] = None
        r["rank"] = None
    key = "composite" if by == "composite" else by
    vals = _column(scored, key)
    sign = -1.0 if key in LOWER_BETTER else 1.0
    # stable sort: best first, NaN last, ties keep input order
    score = np.where(np.isfinite(vals), sign * vals, -np.inf)
    order = sorted(range(len(scored)), key=lambda k: (-score[k], k))
    ranked = [scored[k] for k in order]
    for pos, r in enumerate(ranked, start=1):
        r["rank"] = pos
    return ranked + unscored


def pareto_front(rows: Sequence[Mapping[str, Any]],
                 objectives: Sequence[str] | Mapping[str, str]) -> list[Mapping[str, Any]]:
    """Non-dominated rows (original order) for the given objectives.

    ``objectives`` is a list of metric names (direction inferred: LOWER_BETTER metrics are
    minimised, everything else maximised) or a mapping name -> "max" | "min". Rows missing
    any objective are excluded.
    """
    if isinstance(objectives, Mapping):
        names = list(objectives)
        signs = np.array([1.0 if str(objectives[k]).lower().startswith("max") else -1.0 for k in names])
    else:
        names = list(objectives)
        signs = np.array([_direction(k) for k in names])
    if not rows or not names:
        return list(rows)
    M = np.stack([_column(rows, k) for k in names], axis=1) * signs  # maximise all
    ok = np.isfinite(M).all(axis=1)
    idx = np.flatnonzero(ok)
    V = M[idx]
    ge = (V[:, None, :] >= V[None, :, :]).all(axis=2)
    gt = (V[:, None, :] > V[None, :, :]).any(axis=2)
    dominated = (ge & gt).any(axis=0)  # column j dominated by some row i
    return [rows[k] for k in idx[~dominated]]
