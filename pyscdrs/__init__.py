"""pyscdrs: clean pure-Python reimplementation of scDRS.

A standalone, numerically-faithful reimplementation of **scDRS**
(single-cell disease-relevance score; Zhang*, Hou*, et al. & Price,
*Nature Genetics* 2022, 54:1572-1580) for the omicverse project.

scDRS scores individual cells for their relevance to a disease /
complex trait, using a polygenic gene set derived from GWAS (typically
via MAGMA).  The score is a covariate-corrected, technical-variance
weighted average of disease-gene expression, calibrated against
Monte-Carlo control gene sets matched on the gene-level
mean-variance relationship.

Pipeline
--------
1. :func:`preprocess` — covariate correction + per-gene / per-cell
   statistics + mean-variance bins, stored in ``adata.uns["SCDRS_PARAM"]``.
2. :func:`score_cell` — raw score, Monte-Carlo control gene sets,
   normalized score, per-cell Monte-Carlo and pooled empirical p-values.
3. Downstream — :func:`downstream_group_analysis`,
   :func:`downstream_corr_analysis`, :func:`downstream_gene_analysis`.

Gene-set / data I/O
-------------------
* :func:`load_gs` / :func:`save_gs` / :func:`munge_gs` — MAGMA-style
  ``.gs`` gene-set files.
* :func:`load_h5ad` — read + normalize ``.h5ad`` single-cell data.

Quick-start
-----------
>>> import anndata, pandas as pd
>>> import pyscdrs as scdrs
>>> adata = anndata.read_h5ad("toydata_mouse.h5ad")
>>> df_cov = pd.read_csv("toydata_mouse.cov", sep="\\t", index_col=0)
>>> scdrs.preprocess(adata, cov=df_cov)
>>> dict_gs = scdrs.load_gs("toydata_mouse.gs")
>>> genes, weights = dict_gs["toydata_gs_mouse"]
>>> df_res = scdrs.score_cell(adata, genes, gene_weight=weights, n_ctrl=1000)
>>> df_res[["raw_score", "norm_score", "pval", "zscore"]].head()

Numerical parity with the original ``scdrs`` package is the design
goal: with the same ``random_seed`` the control gene sets, scores and
p-values match.
"""
from __future__ import annotations

from .downstream import (
    downstream_corr_analysis,
    downstream_gene_analysis,
    downstream_group_analysis,
    gearys_c,
    test_gearysc,
)
from .io import (
    convert_species_name,
    load_gs,
    load_h5ad,
    load_homolog_mapping,
    munge_gs,
    pval2zsc,
    save_gs,
    zsc2pval,
)
from .preprocess import category2dummy, compute_stats, preprocess, reg_out
from .score import score_cell

__version__ = "0.1.0"

__all__ = [
    # preprocessing
    "preprocess",
    "compute_stats",
    "reg_out",
    "category2dummy",
    # scoring
    "score_cell",
    # downstream
    "downstream_group_analysis",
    "downstream_corr_analysis",
    "downstream_gene_analysis",
    "test_gearysc",
    "gearys_c",
    # gene-set / data I/O
    "load_gs",
    "save_gs",
    "munge_gs",
    "load_h5ad",
    "load_homolog_mapping",
    "convert_species_name",
    "zsc2pval",
    "pval2zsc",
]
