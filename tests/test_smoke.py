"""Pure-Python smoke tests for pyscdrs — no reference package needed.

These run the full pipeline on a tiny synthetic AnnData and check
internal consistency (shapes, dtypes, value ranges, determinism).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

import pyscdrs as scdrs

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_adata():
    """A small synthetic single-cell AnnData (200 cells x 300 genes)."""
    import anndata

    rng = np.random.default_rng(0)
    n_cell, n_gene = 200, 300
    # Poisson counts -> size-factor normalize -> log1p
    counts = rng.poisson(lam=1.0, size=(n_cell, n_gene)).astype(np.float32)
    # Boost a "disease" signal in the first 40 genes for half the cells
    counts[:100, :40] += rng.poisson(lam=3.0, size=(100, 40))
    libsize = counts.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1
    norm = counts / libsize * 1e4
    X = np.log1p(norm).astype(np.float32)

    obs = pd.DataFrame(
        {
            "cell_type": ["A"] * 100 + ["B"] * 100,
            "n_genes": (counts > 0).sum(axis=1),
            "gradient": rng.normal(size=n_cell),
        },
        index=[f"cell{i}" for i in range(n_cell)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_gene)])
    adata = anndata.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    return adata


@pytest.fixture(scope="module")
def disease_genes():
    """The 40 'disease' genes baked into the synthetic data."""
    return [f"gene{i}" for i in range(40)]


# ----------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------
def test_preprocess_normal_mode(synthetic_adata):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    assert "SCDRS_PARAM" in adata.uns
    param = adata.uns["SCDRS_PARAM"]
    assert param["FLAG_SPARSE"] is True
    assert param["FLAG_COV"] is False
    gs = param["GENE_STATS"]
    for col in ["mean", "var", "var_tech", "ct_mean", "ct_var", "mean_var"]:
        assert col in gs.columns
    assert gs.shape[0] == adata.shape[1]
    assert (gs["var"].astype(float) >= 0).all()
    assert (gs["var_tech"].astype(float) >= 0).all()
    cs = param["CELL_STATS"]
    assert cs.shape[0] == adata.shape[0]


def test_preprocess_with_covariate(synthetic_adata):
    adata = synthetic_adata.copy()
    df_cov = pd.DataFrame(
        {"const": 1.0, "n_genes": adata.obs["n_genes"].values},
        index=adata.obs_names,
    )
    scdrs.preprocess(adata, cov=df_cov)
    param = adata.uns["SCDRS_PARAM"]
    assert param["FLAG_COV"] is True
    # sparse + cov -> implicit covariate correction
    assert "COV_MAT" in param and "COV_BETA" in param and "COV_GENE_MEAN" in param


def test_reg_out_residual_orthogonality():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    X[:, 0] = 1.0
    Y = X @ rng.normal(size=(3, 4)) + 0.1 * rng.normal(size=(50, 4))
    resid = scdrs.reg_out(Y, X)
    # residual must be orthogonal to covariates
    assert np.allclose(X.T @ resid, 0, atol=1e-8)


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def test_score_cell_basic(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(adata, disease_genes, n_ctrl=50, random_seed=0)
    assert list(df_res.columns) == [
        "raw_score", "norm_score", "mc_pval", "pval", "nlog10_pval", "zscore",
    ]
    assert df_res.shape[0] == adata.shape[0]
    assert df_res["pval"].between(0, 1).all()
    assert df_res["mc_pval"].between(0, 1).all()
    assert not df_res["norm_score"].isna().any()


def test_score_cell_deterministic(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    r1 = scdrs.score_cell(adata, disease_genes, n_ctrl=30, random_seed=42)
    r2 = scdrs.score_cell(adata, disease_genes, n_ctrl=30, random_seed=42)
    assert np.array_equal(r1["norm_score"].values, r2["norm_score"].values)
    assert np.array_equal(r1["pval"].values, r2["pval"].values)


def test_score_cell_seed_changes_result(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    r1 = scdrs.score_cell(adata, disease_genes, n_ctrl=30, random_seed=0)
    r2 = scdrs.score_cell(adata, disease_genes, n_ctrl=30, random_seed=1)
    # raw scores are seed-independent, normalized p-values are not
    assert np.allclose(r1["raw_score"].values, r2["raw_score"].values)
    assert not np.array_equal(r1["pval"].values, r2["pval"].values)


@pytest.mark.parametrize("weight_opt", ["uniform", "vs", "inv_std", "od"])
def test_score_cell_weight_options(synthetic_adata, disease_genes, weight_opt):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(
        adata, disease_genes, n_ctrl=20, weight_opt=weight_opt, random_seed=0
    )
    assert df_res.shape[0] == adata.shape[0]
    assert df_res["pval"].between(0, 1).all()


def test_score_cell_disease_signal_enriched(synthetic_adata, disease_genes):
    """Cells with the baked-in disease signal should score higher."""
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(adata, disease_genes, n_ctrl=100, random_seed=0)
    signal = df_res["norm_score"].values[:100]
    background = df_res["norm_score"].values[100:]
    assert signal.mean() > background.mean()


def test_score_cell_return_ctrl(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(
        adata, disease_genes, n_ctrl=10, random_seed=0,
        return_ctrl_raw_score=True, return_ctrl_norm_score=True,
    )
    assert "ctrl_raw_score_0" in df_res.columns
    assert "ctrl_norm_score_9" in df_res.columns


# ----------------------------------------------------------------------
# downstream
# ----------------------------------------------------------------------
def test_downstream_gene_analysis(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(adata, disease_genes, n_ctrl=20, random_seed=0)
    df_gene = scdrs.downstream_gene_analysis(adata, df_res)
    assert list(df_gene.columns) == ["CORR", "RANK"]
    assert df_gene.shape[0] == adata.shape[1]
    assert df_gene["RANK"].tolist() == list(range(adata.shape[1]))


def test_downstream_corr_analysis(synthetic_adata, disease_genes):
    adata = synthetic_adata.copy()
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(
        adata, disease_genes, n_ctrl=20, random_seed=0, return_ctrl_norm_score=True
    )
    df_corr = scdrs.downstream_corr_analysis(adata, df_res, var_cols=["gradient"])
    assert list(df_corr.columns) == ["n_ctrl", "corr_mcp", "corr_mcz"]
    assert df_corr.loc["gradient", "corr_mcp"] >= 0


def test_downstream_group_analysis(synthetic_adata, disease_genes):
    import scanpy as sc

    adata = synthetic_adata.copy()
    sc.pp.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    scdrs.preprocess(adata, cov=None)
    df_res = scdrs.score_cell(
        adata, disease_genes, n_ctrl=20, random_seed=0, return_ctrl_norm_score=True
    )
    dict_group = scdrs.downstream_group_analysis(
        adata, df_res, group_cols=["cell_type"]
    )
    df_group = dict_group["cell_type"]
    for col in ["n_cell", "assoc_mcp", "assoc_mcz", "hetero_mcp", "hetero_mcz"]:
        assert col in df_group.columns
    assert set(df_group.index) == {"A", "B"}


# ----------------------------------------------------------------------
# gene-set I/O
# ----------------------------------------------------------------------
def test_load_save_gs_roundtrip(tmp_path):
    gs = {
        "traitA": (["g1", "g2", "g3"], [1.0, 2.0, 0.5]),
        "traitB": (["g4", "g5"], [1.0, 1.0]),
    }
    gs_file = tmp_path / "test.gs"
    scdrs.save_gs(str(gs_file), gs)
    gs_loaded = scdrs.load_gs(str(gs_file))
    assert set(gs_loaded) == set(gs)
    assert gs_loaded["traitA"][0] == ["g1", "g2", "g3"]
    assert np.allclose(gs_loaded["traitA"][1], [1.0, 2.0, 0.5])


def test_load_gs_unweighted(tmp_path):
    gs_file = tmp_path / "uw.gs"
    pd.DataFrame(
        {"TRAIT": ["t1"], "GENESET": ["g1,g2,g3"]}
    ).to_csv(gs_file, sep="\t", index=False)
    gs = scdrs.load_gs(str(gs_file))
    assert gs["t1"][0] == ["g1", "g2", "g3"]
    assert gs["t1"][1] == [1.0, 1.0, 1.0]


def test_load_gs_intersect(tmp_path):
    gs_file = tmp_path / "isec.gs"
    pd.DataFrame(
        {"TRAIT": ["t1"], "GENESET": ["g1,g2,g3,g4"]}
    ).to_csv(gs_file, sep="\t", index=False)
    gs = scdrs.load_gs(str(gs_file), to_intersect=["g2", "g4", "g9"])
    assert set(gs["t1"][0]) == {"g2", "g4"}


def test_munge_gs(tmp_path):
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(500)]
    df_z = pd.DataFrame(
        {"traitX": rng.normal(size=500)}, index=genes
    )
    out = tmp_path / "munged.gs"
    gs = scdrs.munge_gs(str(out), df_zscore=df_z, weight="zscore", n_max=120, n_min=100)
    assert out.exists()
    assert 100 <= len(gs["traitX"][0]) <= 120


def test_zsc_pval_roundtrip():
    z = np.array([0.0, 1.0, 2.5, 4.0])
    p = scdrs.zsc2pval(z)
    z2 = scdrs.pval2zsc(p)
    assert np.allclose(z, z2, atol=1e-6)


def test_convert_species_name():
    assert scdrs.convert_species_name("Mouse") == "mmusculus"
    assert scdrs.convert_species_name("human") == "hsapiens"
    with pytest.raises(ValueError):
        scdrs.convert_species_name("frog")
