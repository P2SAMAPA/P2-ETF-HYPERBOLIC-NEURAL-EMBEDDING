# 🌐 P2-ETF-HYPERBOLIC-NEURAL-EMBEDDING

**Hyperbolic Neural Embedding Engine — Poincaré Disk (Nickel & Kiela 2017)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine embeds ETF return correlations and macro signals into the
**Poincaré disk** (2D hyperbolic space), faithfully representing the latent
hierarchical structure of ETF correlations that Euclidean geometry distorts
exponentially.

From the embedding, it extracts three hierarchy-aware signals: how central
each ETF is in the correlation hierarchy, how close it sits to the macro signal
cluster, and how isolated it is from its peers.

---

## Theory

### Why Hyperbolic Space?

ETF correlations have latent hierarchy:
```
Broad market → Sector → Sub-sector → Individual ETF
```

To represent an N-node tree faithfully:
- **Euclidean ℝᵈ**: requires O(N) dimensions
- **Hyperbolic ℍ²**: requires just **2 dimensions**

### Poincaré Disk Model

ℍ² is represented as the open unit disk D = {z ∈ ℝ² : |z| < 1} with metric:

```
ds² = 4/(1−|z|²)² · (dx² + dy²)
```

Distances grow exponentially toward the boundary — ETFs near origin are
hierarchically central; ETFs near boundary are peripheral/idiosyncratic.

### Poincaré Distance

```
d(u,v) = arccosh(1 + 2|u−v|² / ((1−|u|²)(1−|v|²)))
```

### Riemannian SGD (Nickel & Kiela 2017)

Minimise: L = Σᵢⱼ (d(zᵢ,zⱼ) − target_{ij})²

Target distances: target_{ij} = −log((1 + corr_{ij})/2)
- High correlation → small target distance (nearby in disk)
- Low/negative correlation → large target distance (far in disk)

Riemannian gradient: grad_R = ((1−|z|²)²/4) · grad_E

Update: z ← proj(z − η·grad_R), where proj clips to open disk.

---

## Score Construction

```
score = 0.45·(−|z|)  +  0.35·(−d(z, z_macro))  +  0.20·d_knn
```

| Component | Meaning | Signal |
|-----------|---------|--------|
| −\|z\| (centrality) | Near origin → regime anchor | Positive |
| −d(z, z_macro) | Near macro centre → macro-sensitive | Positive |
| d_knn (isolation) | Far from k neighbours → idiosyncratic | Positive |

---

## Distinction from Other Suite Engines

| Engine | Space | Structure captured |
|--------|-------|-------------------|
| NETWORK-CENTRALITY | Euclidean graph | Degree / betweenness |
| GRAPH-TRANSFORMER | Euclidean | Message passing |
| GEOMETRIC-DEEP-LEARNING | Euclidean manifolds | Differential geometry |
| **HNE (this engine)** | **Hyperbolic ℍ²** | **Hierarchical tree structure** |
| TDA-HOMOLOGY | Topological | Persistent holes |

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-HYPERBOLIC-NEURAL-EMBEDDING/
├── config.py          # Universes, Poincaré disk hyperparameters
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── hne_engine.py      # Core: Poincaré distance, Riemannian SGD, scoring
├── trainer.py         # Orchestrator: load → embed → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (vectorised — very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-HYPERBOLIC-NEURAL-EMBEDDING
cd P2-ETF-HYPERBOLIC-NEURAL-EMBEDDING
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-hyperbolic-embedding-results`

---

## References

- Nickel, M. & Kiela, D. (2017). Poincaré embeddings for learning hierarchical
  representations. *NeurIPS 2017*.
- Ganea, O., Bécigneul, G. & Hofmann, T. (2018). Hyperbolic neural networks.
  *NeurIPS 2018*.
- Chami, I. et al. (2019). Hyperbolic graph convolutional neural networks.
  *NeurIPS 2019*.
- Sarkar, R. (2011). Low distortion Delaunay embedding of trees in hyperbolic
  plane. *Graph Drawing 2011*.
