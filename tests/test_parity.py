"""Numerical-parity tests: pyscdrs vs the original scdrs package.

Both packages are run on scDRS's own bundled toy data (``toydata_mouse``)
with identical ``random_seed``.  The same seed must yield the same
control gene sets and therefore identical scores and p-values.

These tests skip gracefully if the upstream ``scdrs`` package is not
importable.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import pyscdrs

warnings.filterwarnings("ignore")

scdrs = pytest.importorskip("scdrs", reason="reference scdrs package not installed")

_DATA_DIR = os.path.dirname(__file__)
_H5AD = os.path.join(_DATA_DIR, "toydata_mouse.h5ad")
_COV = os.path.join(_DATA_DIR, "toydata_mouse.cov")
_GS = os.path.join(_DATA_DIR, "toydata_mouse.gs")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_H5AD), reason="bundled toy data missing"
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_inputs():
    import anndata

    adata = anndata.read_h5ad(_H5AD)
    df_cov = pd.read_csv(_COV, sep="\t", index_col=0)
    dict_gs = pyscdrs.load_gs(_GS)
    genes, weights = dict_gs["toydata_gs_mouse"]
    return adata, df_cov, genes, weights


# ----------------------------------------------------------------------
# load_gs parity
# ----------------------------------------------------------------------
def test_load_gs_matches_reference():
    gs_mine = pyscdrs.load_gs(_GS)
    gs_ref = scdrs.util.load_gs(_GS)
    assert set(gs_mine) == set(gs_ref)
    for trait in gs_mine:
        assert gs_mine[trait][0] == gs_ref[trait][0]
        assert np.allclose(gs_mine[trait][1], gs_ref[trait][1])


# ----------------------------------------------------------------------
# preprocess parity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("use_cov", [False, True])
def test_preprocess_matches_reference(toy_inputs, use_cov):
    adata, df_cov, _, _ = toy_inputs
    cov = df_cov if use_cov else None

    a_mine = adata.copy()
    a_ref = adata.copy()
    pyscdrs.preprocess(a_mine, cov=cov)
    scdrs.pp.preprocess(a_ref, cov=cov)

    gs_mine = a_mine.uns["SCDRS_PARAM"]["GENE_STATS"]
    gs_ref = a_ref.uns["SCDRS_PARAM"]["GENE_STATS"]

    for col in ["mean", "var", "var_tech", "ct_mean", "ct_var"]:
        np.testing.assert_allclose(
            gs_mine[col].astype(float).values,
            gs_ref[col].astype(float).values,
            rtol=1e-6, atol=1e-8,
            err_msg=f"GENE_STATS['{col}'] mismatch (use_cov={use_cov})",
        )
    # mean-variance bins must match exactly
    assert (gs_mine["mean_var"] == gs_ref["mean_var"]).all(), "mean_var bins differ"

    cs_mine = a_mine.uns["SCDRS_PARAM"]["CELL_STATS"]
    cs_ref = a_ref.uns["SCDRS_PARAM"]["CELL_STATS"]
    for col in ["mean", "var"]:
        np.testing.assert_allclose(
            cs_mine[col].astype(float).values,
            cs_ref[col].astype(float).values,
            rtol=1e-6, atol=1e-8,
        )


# ----------------------------------------------------------------------
# score_cell parity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("use_cov", [False, True])
def test_score_cell_matches_reference(toy_inputs, use_cov):
    adata, df_cov, genes, weights = toy_inputs
    cov = df_cov if use_cov else None

    a_mine = adata.copy()
    a_ref = adata.copy()
    pyscdrs.preprocess(a_mine, cov=cov)
    scdrs.pp.preprocess(a_ref, cov=cov)

    r_mine = pyscdrs.score_cell(
        a_mine, genes, gene_weight=weights, n_ctrl=100, random_seed=0
    )
    r_ref = scdrs.score_cell(
        a_ref, genes, gene_weight=weights, n_ctrl=100, random_seed=0
    )

    # raw / norm score: Pearson r essentially 1
    for col in ["raw_score", "norm_score"]:
        r = np.corrcoef(r_mine[col].values, r_ref[col].values)[0, 1]
        assert r > 0.9999, f"{col} Pearson r={r} (use_cov={use_cov})"

    # p-values: same seed -> same MC null -> near-identical
    for col in ["pval", "mc_pval", "zscore"]:
        np.testing.assert_allclose(
            r_mine[col].values, r_ref[col].values, rtol=1e-3, atol=1e-4,
            err_msg=f"{col} mismatch (use_cov={use_cov})",
        )


@pytest.mark.parametrize("weight_opt", ["uniform", "vs", "inv_std"])
def test_score_cell_weight_opt_parity(toy_inputs, weight_opt):
    adata, _, genes, weights = toy_inputs
    a_mine = adata.copy()
    a_ref = adata.copy()
    pyscdrs.preprocess(a_mine, cov=None)
    scdrs.pp.preprocess(a_ref, cov=None)

    r_mine = pyscdrs.score_cell(
        a_mine, genes, gene_weight=weights, n_ctrl=50,
        weight_opt=weight_opt, random_seed=0,
    )
    r_ref = scdrs.score_cell(
        a_ref, genes, gene_weight=weights, n_ctrl=50,
        weight_opt=weight_opt, random_seed=0,
    )
    r = np.corrcoef(r_mine["norm_score"].values, r_ref["norm_score"].values)[0, 1]
    assert r > 0.9999, f"weight_opt={weight_opt}: norm_score r={r}"
    np.testing.assert_allclose(
        r_mine["pval"].values, r_ref["pval"].values, rtol=1e-3, atol=1e-4
    )


def test_score_cell_full_score_parity(toy_inputs):
    """Control normalized scores must also match."""
    adata, df_cov, genes, weights = toy_inputs
    a_mine = adata.copy()
    a_ref = adata.copy()
    pyscdrs.preprocess(a_mine, cov=df_cov)
    scdrs.pp.preprocess(a_ref, cov=df_cov)

    r_mine = pyscdrs.score_cell(
        a_mine, genes, gene_weight=weights, n_ctrl=50, random_seed=0,
        return_ctrl_norm_score=True,
    )
    r_ref = scdrs.score_cell(
        a_ref, genes, gene_weight=weights, n_ctrl=50, random_seed=0,
        return_ctrl_norm_score=True,
    )
    for i in [0, 10, 49]:
        col = f"ctrl_norm_score_{i}"
        np.testing.assert_allclose(
            r_mine[col].values, r_ref[col].values, rtol=1e-3, atol=1e-4,
            err_msg=f"{col} mismatch",
        )


# ----------------------------------------------------------------------
# downstream parity
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored_pair(toy_inputs):
    import scanpy as sc

    adata, df_cov, genes, weights = toy_inputs
    a_mine = adata.copy()
    a_ref = adata.copy()
    pyscdrs.preprocess(a_mine, cov=df_cov)
    scdrs.pp.preprocess(a_ref, cov=df_cov)
    r_mine = pyscdrs.score_cell(
        a_mine, genes, gene_weight=weights, n_ctrl=100, random_seed=0,
        return_ctrl_norm_score=True,
    )
    r_ref = scdrs.score_cell(
        a_ref, genes, gene_weight=weights, n_ctrl=100, random_seed=0,
        return_ctrl_norm_score=True,
    )
    # shared neighbor graph for the heterogeneity test
    sc.pp.neighbors(a_mine, n_neighbors=15, n_pcs=20)
    a_ref.obsp = a_mine.obsp
    a_ref.uns["neighbors"] = a_mine.uns.get("neighbors", {})
    return a_mine, a_ref, r_mine, r_ref


def test_downstream_group_analysis_parity(scored_pair):
    a_mine, a_ref, r_mine, r_ref = scored_pair
    g_mine = pyscdrs.downstream_group_analysis(
        a_mine, r_mine, group_cols=["cell_type"]
    )["cell_type"]
    g_ref = scdrs.method.downstream_group_analysis(
        a_ref, r_ref, group_cols=["cell_type"]
    )["cell_type"]
    for col in ["assoc_mcp", "assoc_mcz", "hetero_mcp", "hetero_mcz", "n_fdr_0.1"]:
        np.testing.assert_allclose(
            g_mine[col].astype(float).values,
            g_ref[col].astype(float).values,
            rtol=1e-3, atol=1e-4,
            err_msg=f"group analysis '{col}' mismatch",
        )


def test_downstream_gene_analysis_parity(scored_pair):
    a_mine, a_ref, r_mine, r_ref = scored_pair
    gene_mine = pyscdrs.downstream_gene_analysis(a_mine, r_mine)
    gene_ref = scdrs.method.downstream_gene_analysis(a_ref, r_ref)
    np.testing.assert_allclose(
        gene_mine.loc[gene_ref.index, "CORR"].astype(float).values,
        gene_ref["CORR"].astype(float).values,
        rtol=1e-4, atol=1e-5,
    )


def test_downstream_corr_analysis_parity(scored_pair):
    a_mine, a_ref, r_mine, r_ref = scored_pair
    c_mine = pyscdrs.downstream_corr_analysis(
        a_mine, r_mine, var_cols=["causal_variable"]
    )
    c_ref = scdrs.method.downstream_corr_analysis(
        a_ref, r_ref, var_cols=["causal_variable"]
    )
    for col in ["corr_mcp", "corr_mcz"]:
        np.testing.assert_allclose(
            c_mine[col].astype(float).values,
            c_ref[col].astype(float).values,
            rtol=1e-3, atol=1e-4,
        )
