"""
hne_engine.py — Hyperbolic Neural Embedding Engine (Poincaré Disk)
===================================================================

Theory
------
**Why hyperbolic space for ETF correlations?**

ETF return correlations often have a latent hierarchical structure:
  broad market → sector → sub-sector → individual ETF

Euclidean geometry distorts hierarchical distances exponentially. To represent
an N-node tree faithfully in Euclidean space requires O(N) dimensions;
in hyperbolic space it requires just 2 dimensions.

The Poincaré disk model of 2D hyperbolic space represents all of ℍ² as
the open unit disk D = {z ∈ ℝ² : |z| < 1}, with metric:

    ds² = 4/(1−|z|²)² · (dx² + dy²)

Key property: **distances grow exponentially toward the boundary**. This
naturally represents hierarchies — parent nodes sit near the origin, child
nodes sit near the boundary, and the boundary itself (|z|=1) is at
infinite distance from any interior point.

**Poincaré Disk Distance**

For two points u, v ∈ D:

    d(u, v) = arccosh(1 + 2|u−v|² / ((1−|u|²)(1−|v|²)))

**Poincaré Disk Embedding (Nickel & Kiela 2017)**

Given a similarity matrix S_{ij} (correlation of ETF returns), we want
embeddings z_1, ..., z_N ∈ D such that:

    d(z_i, z_j) ≈ f(1 − S_{ij})    (dissimilar → far apart in disk)

where f converts similarity to target distance:
    target_{ij} = -log((1 + S_{ij})/2 + ε)

Training: Riemannian SGD minimising:
    L = Σ_{i≠j} (d(z_i, z_j) − target_{ij})²

**Riemannian SGD on the Poincaré Disk**

The Riemannian gradient on ℍ² is the Euclidean gradient scaled by the
inverse of the Riemannian metric:

    grad_R = ((1−|z|²)²/4) · grad_E

The update step:

    z ← Exp_z(−η · grad_R)

For small steps, approximated as:
    z ← proj(z − η · grad_R)

where proj(z) = z / max(|z|, 1−ε) clips to the open unit disk.

**Score Construction**

After embedding all ETFs:

1. **Centrality** (−|z_i|): ETFs near the origin are central to the
   correlation hierarchy (regime anchors). Peripheral ETFs (|z_i| → 1)
   are isolated/idiosyncratic. Central ETFs in risk-on periods → positive.

2. **Macro proximity**: embed macro signals as additional nodes.
   Low hyperbolic distance from ETF to macro centre → macro-sensitive.
   Direction of sensitivity relative to current macro regime → sign of signal.

3. **Cluster score**: mean hyperbolic distance to k nearest neighbours.
   High isolation in disk → idiosyncratic return potential.

Composite score = weighted blend, cross-sectionally z-scored per universe/window.

**Distinct from GEOMETRIC-DEEP-LEARNING (in suite):**
  - GDL: graph neural networks on Euclidean manifolds
  - HNE: embedding into hyperbolic space to faithfully represent hierarchy
    The key difference is the *geometry of the embedding space*

References
----------
- Nickel, M. & Kiela, D. (2017). Poincaré embeddings for learning hierarchical
  representations. NeurIPS 2017.
- Ganea, O., Bécigneul, G. & Hofmann, T. (2018). Hyperbolic neural networks.
  NeurIPS 2018.
- Chami, I. et al. (2019). Hyperbolic graph convolutional neural networks.
  NeurIPS 2019.
- Sarkar, R. (2011). Low distortion Delaunay embedding of trees in hyperbolic
  plane. Graph Drawing 2011.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Poincaré disk geometry ────────────────────────────────────────────────────

def _poincare_dist_matrix(Z: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Vectorised pairwise Poincaré disk distances.
    Z: (N, d) — points in open unit disk
    Returns: (N, N) symmetric distance matrix
    """
    N  = len(Z)
    sq = np.sum(Z**2, axis=1)                               # (N,) |z_i|²
    sq = np.clip(sq, 0, 1 - config.HNE_EPS)

    # |u - v|² for all pairs
    U  = Z[:, None, :]                                       # (N,1,d)
    V  = Z[None, :, :]                                       # (1,N,d)
    diff_sq = np.sum((U - V)**2, axis=-1)                    # (N,N)

    # (1 - |u|²)(1 - |v|²)
    denom = (1 - sq[:, None]) * (1 - sq[None, :])           # (N,N)
    denom = np.clip(denom, eps, None)

    arg = 1.0 + 2.0 * diff_sq / denom
    arg = np.clip(arg, 1.0 + eps, None)
    return np.arccosh(arg)


def _poincare_dist_vec(z: np.ndarray, Z: np.ndarray,
                        eps: float = 1e-7) -> np.ndarray:
    """Distance from point z to all rows of Z."""
    sq_z = np.clip(np.sum(z**2), 0, 1-config.HNE_EPS)
    sq_Z = np.clip(np.sum(Z**2, axis=1), 0, 1-config.HNE_EPS)
    diff_sq = np.sum((z - Z)**2, axis=1)
    denom   = (1 - sq_z) * (1 - sq_Z)
    denom   = np.clip(denom, eps, None)
    arg     = 1.0 + 2.0 * diff_sq / denom
    return np.arccosh(np.clip(arg, 1.0+eps, None))


def _proj(Z: np.ndarray) -> np.ndarray:
    """Project all rows of Z back into open unit disk."""
    norms = np.linalg.norm(Z, axis=-1, keepdims=True)
    scale = np.where(norms >= 1.0, (1.0 - config.HNE_EPS) / norms, 1.0)
    return Z * scale


def _riemannian_grad(grad_e: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Convert Euclidean gradient to Riemannian gradient on Poincaré disk.
    grad_R = ((1 - |z|²)² / 4) · grad_E
    """
    sq    = np.clip(np.sum(z**2), 0, 1-config.HNE_EPS)
    scale = ((1.0 - sq) ** 2) / 4.0
    return scale * grad_e


# ── Similarity → target distance ──────────────────────────────────────────────

def _corr_to_target_dist(corr: np.ndarray) -> np.ndarray:
    """
    Map correlation matrix to target hyperbolic distances.
    High correlation (≈1) → target distance ≈ 0 (nearby)
    Low/negative correlation (≈−1) → large target distance
    """
    # Shift corr to [0,1], then map to distance
    sim = np.clip((1.0 + corr) / 2.0, config.HNE_DIST_EPS, 1.0)
    return -np.log(sim)


# ── Poincaré embedding via Riemannian SGD ─────────────────────────────────────

def _fit_poincare(corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Fit Poincaré disk embeddings for N entities with similarity matrix `corr`.

    Parameters
    ----------
    corr : (N, N) correlation matrix
    rng  : random generator

    Returns
    -------
    Z : (N, 2) embedding in Poincaré disk
    """
    N = len(corr)
    d = config.EMBED_DIM

    # Initialise near origin
    Z = rng.normal(0, 0.01, (N, d))
    Z = _proj(Z)

    targets = _corr_to_target_dist(corr)   # (N, N)
    np.fill_diagonal(targets, 0.0)

    for epoch in range(config.HNE_EPOCHS):
        lr = config.HNE_LR_BURN if epoch < config.HNE_BURN_IN else config.HNE_LR

        # Vectorised pairwise distances
        D  = _poincare_dist_matrix(Z)      # (N, N)

        # Loss gradient w.r.t. distances: dL/dD_{ij} = 2(D_{ij} - T_{ij})
        dL_dD = 2.0 * (D - targets)        # (N, N)

        # Gradient of d(u,v) w.r.t. u (using autodiff by hand)
        # d(u,v) = arccosh(alpha),  alpha = 1 + 2|u-v|²/((1-|u|²)(1-|v|²))
        # dα/du = 4(u-v)/((1-|u|²)(1-|v|²)) + 4|u-v|²·u/(1-|u|²)²(1-|v|²)
        # dd/du = dα/du / sqrt(alpha²-1)

        sq_u = np.clip(np.sum(Z**2, axis=1), 0, 1-config.HNE_EPS)  # (N,)
        sq_v = sq_u.copy()

        diff   = Z[:, None, :] - Z[None, :, :]   # (N,N,d) u-v
        dsq    = np.sum(diff**2, axis=-1)         # (N,N)
        den_uv = (1-sq_u[:,None]) * (1-sq_v[None,:])  # (N,N)
        alpha  = 1 + 2*dsq / np.clip(den_uv, 1e-7, None)  # (N,N)

        # Gradient of arccosh(alpha): 1/sqrt(alpha²-1)
        darch  = 1.0 / np.sqrt(np.clip(alpha**2 - 1, 1e-7, None))  # (N,N)

        # dα/du[i,j,:] = 4*(Z[i]-Z[j])/den_uv[i,j] + 4*dsq[i,j]*Z[i]/(1-|Z[i]|²)*den_uv[i,j]
        term1  = 4 * diff / np.clip(den_uv[:,:,None], 1e-7, None)
        term2  = 4 * dsq[:,:,None] * Z[:,None,:] / np.clip(
                    (1-sq_u[:,None,None]) * den_uv[:,:,None], 1e-7, None)
        dalpha = term1 + term2                    # (N,N,d)

        # Full Euclidean gradient for each point i
        chain  = (dL_dD * darch)[:,:,None]       # (N,N,1)
        grad_e = (chain * dalpha).sum(axis=1)     # (N,d)

        # Convert to Riemannian gradient and update
        for i in range(N):
            grad_r = _riemannian_grad(grad_e[i], Z[i])
            Z[i]   = Z[i] - lr * grad_r

        Z = _proj(Z)

    return Z


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_hne_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Embed ETF returns + macro signals into the Poincaré disk and compute
    hierarchy-aware scores.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite HNE z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail or len(avail) < 3:
        return pd.Series(dtype=float)

    if len(prices) < window + 5:
        return pd.Series(dtype=float)

    # Align macro
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    # ── Compute log returns over window ──────────────────────────────────────
    ret_dict = {}
    for ticker in avail:
        ps = prices_a[ticker].dropna()
        if len(ps) < window + 2:
            continue
        lr = np.log(ps / ps.shift(1)).dropna().values[-window:]
        if not np.isnan(lr).any():
            ret_dict[ticker] = lr

    valid_tickers = list(ret_dict.keys())
    if len(valid_tickers) < 3:
        return pd.Series(dtype=float)

    # ── Build ETF correlation matrix ──────────────────────────────────────────
    R = np.column_stack([ret_dict[t] for t in valid_tickers])  # (window, N)
    corr_etf = np.corrcoef(R.T)                                 # (N, N)
    corr_etf = np.clip(corr_etf, -0.999, 0.999)
    np.fill_diagonal(corr_etf, 1.0)

    # ── Build macro correlation rows ──────────────────────────────────────────
    macro_cols = [c for c in config.MACRO_COLS_CORE if c in macro_a.columns]
    macro_vals = macro_a[macro_cols].values.astype(np.float64) if macro_cols else None

    if macro_vals is not None and len(macro_vals) >= window:
        mac_win = macro_vals[-window:]
        mac_chg = np.diff(mac_win, axis=0, prepend=mac_win[:1])

        # Correlation of each ETF with each macro signal
        corr_etf_macro = np.zeros((len(valid_tickers), len(macro_cols)))
        for j, col_idx in enumerate(range(len(macro_cols))):
            mac_col = mac_chg[:, col_idx]
            for i, ticker in enumerate(valid_tickers):
                if mac_col.std() < 1e-10:
                    continue
                r   = ret_dict[ticker]
                c   = np.corrcoef(r, mac_col)[0, 1]
                corr_etf_macro[i, j] = np.clip(c, -0.999, 0.999)

        # Build joint correlation matrix: ETFs + macro nodes
        N_etf  = len(valid_tickers)
        N_mac  = len(macro_cols)
        N_total = N_etf + N_mac
        corr_joint = np.eye(N_total)
        corr_joint[:N_etf, :N_etf]   = corr_etf
        corr_joint[:N_etf, N_etf:]   = corr_etf_macro
        corr_joint[N_etf:, :N_etf]   = corr_etf_macro.T
        # Macro-macro correlation
        if N_mac > 1:
            mac_corr = np.corrcoef(mac_chg.T)
            corr_joint[N_etf:, N_etf:] = np.clip(mac_corr, -0.999, 0.999)
    else:
        corr_joint   = corr_etf
        N_etf        = len(valid_tickers)
        N_mac        = 0

    # ── Fit Poincaré embedding ────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    print(f"    Embedding {N_etf} ETFs + {N_mac} macro nodes "
          f"(window={window}d, epochs={config.HNE_EPOCHS})")

    try:
        Z = _fit_poincare(corr_joint, rng)    # (N_total, 2)
    except Exception as e:
        print(f"    Embedding failed: {e}")
        return pd.Series(dtype=float)

    Z_etf = Z[:N_etf]     # ETF embeddings
    Z_mac = Z[N_etf:] if N_mac > 0 else None

    # ── Score computation ─────────────────────────────────────────────────────
    # 1. Centrality: negative radial distance (near origin = positive)
    radii = np.linalg.norm(Z_etf, axis=1)          # (N_etf,)

    # 2. Macro proximity: hyperbolic distance to macro centroid
    if Z_mac is not None and len(Z_mac) > 0:
        mac_centre = Z_mac.mean(axis=0, keepdims=True)
        mac_centre = _proj(mac_centre)
        macro_dists = _poincare_dist_vec(mac_centre[0], Z_etf)
    else:
        macro_dists = np.zeros(N_etf)

    # 3. Cluster isolation: mean dist to K nearest neighbours
    D_all = _poincare_dist_matrix(Z_etf)
    np.fill_diagonal(D_all, np.inf)
    K     = min(config.K_NEIGHBOURS, N_etf - 1)
    knn_dists = np.sort(D_all, axis=1)[:, :K].mean(axis=1)  # (N_etf,)

    raw_scores = {}
    for i, ticker in enumerate(valid_tickers):
        s_central = -radii[i]              # near origin → positive
        s_macro   = -macro_dists[i]        # near macro → positive
        s_cluster = knn_dists[i]           # isolated → positive (contrarian)

        composite = (
            config.WEIGHT_CENTRALITY    * s_central
            + config.WEIGHT_MACRO_PROX  * s_macro
            + config.WEIGHT_CLUSTER     * s_cluster
        )
        raw_scores[ticker] = composite
        print(f"    {ticker}: r={radii[i]:.3f}  "
              f"mac_dist={macro_dists[i]:.3f}  "
              f"knn={knn_dists[i]:.3f}  score={composite:.4f}")

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
