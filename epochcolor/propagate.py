"""Edge-aware spreading of painted hints.

Given a base chroma (from a model, or zero) and painted hints, solve for a
correction that matches the hints where you painted, stays smooth inside
regions of similar brightness, and stops at edges. A small pull toward zero
everywhere keeps a fix on a coat from bleeding across the whole frame; the
"spread" setting controls how far it can travel through flat areas.

This is a screened, edge-weighted Poisson problem, the same family as Levin
et al. 2004 "Colorization using Optimization". The matrix is symmetric
positive definite, so a sparse direct solve is exact and fast at the reduced
working size.
"""

from __future__ import annotations

import numpy as np

from .hints import Hints


def _affinities(L: np.ndarray, sigma: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    dx = np.diff(L, axis=1)
    dy = np.diff(L, axis=0)
    if sigma is None:
        g = np.concatenate([np.abs(dx).ravel(), np.abs(dy).ravel()])
        sigma = float(np.clip(2.0 * np.median(g), 0.4, 6.0))
    wx = np.exp(-(dx * dx) / (2.0 * sigma * sigma)) + 1e-4
    wy = np.exp(-(dy * dy) / (2.0 * sigma * sigma)) + 1e-4
    return wx.astype(np.float64), wy.astype(np.float64), sigma


def edge_laplacian(L: np.ndarray, sigma: float | None = None):
    import scipy.sparse as sp

    h, w = L.shape
    n = h * w
    wx, wy, _ = _affinities(L.astype(np.float64), sigma)
    idx = np.arange(n).reshape(h, w)
    rows = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    cols = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    vals = np.concatenate([wx.ravel(), wy.ravel()])
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(deg) - W


def propagate(
    L: np.ndarray,
    base_ab: np.ndarray,
    hints: Hints,
    spread: float = 0.15,
    hint_weight: float = 1e3,
    sigma: float | None = None,
) -> np.ndarray:
    """Return base_ab corrected so it agrees with the hints.

    L, base_ab and hints are all at the same (working) size. spread is the
    distance a correction carries through a flat area, as a fraction of the
    short side. Edges stop it much sooner.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if hints.count == 0:
        return base_ab
    h, w = L.shape
    n = h * w
    lap = edge_laplacian(L, sigma)
    reach = max(1.0, spread * min(h, w))
    lam = 1.0 / (reach * reach)
    m = (hints.mask.ravel() > 0.5).astype(np.float64)
    A = (lap + sp.diags(lam + hint_weight * m)).tocsc()
    solve = spla.factorized(A)
    out = base_ab.astype(np.float64).copy()
    for c in range(2):
        d = (hints.ab[..., c] - base_ab[..., c]).ravel().astype(np.float64)
        x = solve(hint_weight * m * d)
        out[..., c] += x.reshape(h, w)
    return out.astype(np.float32)
