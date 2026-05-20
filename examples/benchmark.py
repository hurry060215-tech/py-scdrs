"""Benchmark: pyscdrs vs the original scdrs package.

Runs ``preprocess`` + ``score_cell`` with both packages on scDRS's
bundled toy data, reports timing and numerical agreement.

Usage
-----
    python examples/benchmark.py
"""
from __future__ import annotations

import os
import time
import warnings

import anndata
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pyscdrs

try:
    import scdrs as _scdrs_ref

    HAS_REF = True
except ImportError:  # pragma: no cover
    HAS_REF = False

_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "tests")
H5AD = os.path.join(_DATA, "toydata_mouse.h5ad")
COV = os.path.join(_DATA, "toydata_mouse.cov")
GS = os.path.join(_DATA, "toydata_mouse.gs")

N_CTRL = 1000
SEED = 0


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    adata = anndata.read_h5ad(H5AD)
    df_cov = pd.read_csv(COV, sep="\t", index_col=0)
    genes, weights = pyscdrs.load_gs(GS)["toydata_gs_mouse"]
    print(f"toy data: {adata.shape[0]} cells x {adata.shape[1]} genes, "
          f"{len(genes)} disease genes, n_ctrl={N_CTRL}")

    # pyscdrs
    a_mine = adata.copy()
    _, t_pp = _time(lambda: pyscdrs.preprocess(a_mine, cov=df_cov))
    r_mine, t_sc = _time(
        lambda: pyscdrs.score_cell(
            a_mine, genes, gene_weight=weights, n_ctrl=N_CTRL, random_seed=SEED
        )
    )
    print(f"\npyscdrs : preprocess {t_pp:.3f}s | score_cell {t_sc:.3f}s")

    if not HAS_REF:
        print("\nreference scdrs not installed; skipping comparison")
        return

    a_ref = adata.copy()
    _, t_pp_r = _time(lambda: _scdrs_ref.pp.preprocess(a_ref, cov=df_cov))
    r_ref, t_sc_r = _time(
        lambda: _scdrs_ref.score_cell(
            a_ref, genes, gene_weight=weights, n_ctrl=N_CTRL, random_seed=SEED
        )
    )
    print(f"scdrs   : preprocess {t_pp_r:.3f}s | score_cell {t_sc_r:.3f}s")

    print("\nnumerical agreement (same random_seed):")
    for col in ["raw_score", "norm_score", "pval", "mc_pval", "zscore"]:
        r = np.corrcoef(r_mine[col].values, r_ref[col].values)[0, 1]
        md = np.abs(r_mine[col].values - r_ref[col].values).max()
        print(f"  {col:12s} Pearson r = {r:.6f}   max|diff| = {md:.3e}")


if __name__ == "__main__":
    main()
