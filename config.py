import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-hyperbolic-embedding-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Poincaré disk embedding hyperparameters ───────────────────────────────────
# Embedding dimension: 2D for Poincaré disk, or higher for Poincaré ball
EMBED_DIM = 2

# Riemannian SGD hyperparameters
HNE_EPOCHS  = 300
HNE_LR      = 0.05        # Riemannian learning rate (larger than Euclidean)
HNE_BURN_IN = 50          # initial epochs with smaller lr for stability
HNE_LR_BURN = 0.001       # burn-in learning rate

# Numerical stability: keep embeddings inside open unit disk
HNE_EPS     = 1e-5        # boundary buffer: max norm = 1 - HNE_EPS

# Similarity → target distance mapping
# We use: target_dist = -log((1 + corr)/2 + eps)
# High correlation (corr→1) → target_dist→0 (nearby in disk)
# Low/negative corr (corr→-1) → target_dist→large (far in disk)
HNE_DIST_EPS = 1e-3

# ── Score construction ────────────────────────────────────────────────────────
# Three Poincaré disk signals:
#
#   centrality_score : radial distance from disk origin |z|
#                      ETFs near origin → central / regime anchor → positive
#                      ETFs near boundary → peripheral / idiosyncratic → negative
#
#   macro_proximity  : hyperbolic distance from ETF to macro centre
#                      Low distance → ETF closely tracks macro → positive
#                      High distance → ETF decoupled from macro → regime signal
#
#   cluster_score    : mean hyperbolic distance to nearest neighbours
#                      Low → tightly clustered with similar ETFs → less alpha
#                      High → isolated in disk → idiosyncratic alpha

WEIGHT_CENTRALITY    = 0.45
WEIGHT_MACRO_PROX    = 0.35
WEIGHT_CLUSTER       = 0.20

# Number of nearest neighbours for cluster score
K_NEIGHBOURS = 3

TOP_N = 3
