"""Downstream analyses for scDRS results.

Clean reimplementation of the ``scdrs.method.downstream_*`` functions:

* :func:`downstream_group_analysis` — cell-group association +
  heterogeneity (Geary's C).
* :func:`downstream_corr_analysis` — correlation of disease score with a
  continuous cell-level variable.
* :func:`downstream_gene_analysis` — genes correlated with the disease
  score.

All take a ``.full_score`` style DataFrame (the ``score_cell`` output
with ``return_ctrl_norm_score=True``).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

__all__ = [
    "downstream_group_analysis",
    "downstream_corr_analysis",
    "downstream_gene_analysis",
    "gearys_c",
    "test_gearysc",
]


# ----------------------------------------------------------------------
# Group-level analysis
# ----------------------------------------------------------------------
def downstream_group_analysis(
    adata,
    df_full_score: pd.DataFrame,
    group_cols: List[str],
    fdr_thresholds: List[float] = [0.05, 0.1, 0.2],
) -> Dict[str, pd.DataFrame]:
    """scDRS group-level analysis.

    For each annotation in ``group_cols`` and each cell group, compute the
    proportion of significant cells, the group-level trait association
    (Monte-Carlo test on the 95th-percentile score) and the group-level
    heterogeneity (Geary's C on the cell-cell graph).

    ``adata.obsp["connectivities"]`` is required (run
    ``scanpy.pp.neighbors`` first).

    Parameters
    ----------
    adata
        AnnData with a ``connectivities`` graph in ``.obsp``.
    df_full_score
        ``score_cell`` output with ``return_ctrl_norm_score=True``.
    group_cols
        Columns of ``adata.obs`` defining cell groups.
    fdr_thresholds
        FDR cutoffs for the significant-cell proportions.

    Returns
    -------
    dict of pandas.DataFrame
        ``group_col -> df_res`` with per-group statistics.
    """
    assert "connectivities" in adata.obsp, (
        "Expect `connectivities` in `adata.obsp`; run `sc.pp.neighbors` first"
    )
    assert len(set(group_cols) - set(adata.obs)) == 0, (
        "Missing `group_cols` variables from `adata.obs.columns`."
    )

    cell_list = sorted(set(adata.obs_names) & set(df_full_score.index))
    control_list = [x for x in df_full_score.columns if x.startswith("ctrl_norm_score")]
    n_ctrl = len(control_list)
    df_reg = adata.obs.loc[cell_list, group_cols].copy()
    df_reg = df_reg.join(
        df_full_score.loc[cell_list, ["norm_score"] + control_list + ["pval"]]
    )

    dict_df_res: Dict[str, pd.DataFrame] = {}
    for group_col in group_cols:
        group_list = sorted(set(adata.obs[group_col]))
        res_cols = [
            "n_cell", "n_ctrl", "assoc_mcp", "assoc_mcz", "hetero_mcp", "hetero_mcz",
        ]
        for fdr_threshold in fdr_thresholds:
            res_cols.append(f"n_fdr_{fdr_threshold}")

        df_res = pd.DataFrame(index=group_list, columns=res_cols, dtype=np.float32)
        df_res.index.name = "group"

        df_fdr = pd.DataFrame(
            {"fdr": multipletests(df_reg["pval"].values, method="fdr_bh")[1]},
            index=df_reg.index,
        )

        for group in group_list:
            group_cell_list = list(df_reg.index[df_reg[group_col] == group])
            df_res.loc[group, ["n_cell", "n_ctrl"]] = [len(group_cell_list), n_ctrl]
            for fdr_threshold in fdr_thresholds:
                df_res.loc[group, f"n_fdr_{fdr_threshold}"] = (
                    df_fdr.loc[group_cell_list, "fdr"].values < fdr_threshold
                ).sum()

        # Association
        for group in group_list:
            group_cell_list = list(df_reg.index[df_reg[group_col] == group])
            score_q95 = np.quantile(df_reg.loc[group_cell_list, "norm_score"], 0.95)
            v_ctrl_score_q95 = np.quantile(
                df_reg.loc[group_cell_list, control_list], 0.95, axis=0
            )
            mc_p = ((v_ctrl_score_q95 >= score_q95).sum() + 1) / (
                v_ctrl_score_q95.shape[0] + 1
            )
            mc_z = (score_q95 - v_ctrl_score_q95.mean()) / v_ctrl_score_q95.std()
            df_res.loc[group, ["assoc_mcp", "assoc_mcz"]] = [mc_p, mc_z]

        # Heterogeneity
        df_rls = test_gearysc(
            adata[cell_list], df_reg.loc[cell_list, :], groupby=group_col
        )
        for ct in group_list:
            mc_p, mc_z = df_rls.loc[ct, ["pval", "zsc"]]
            df_res.loc[ct, ["hetero_mcp", "hetero_mcz"]] = [mc_p, mc_z]

        dict_df_res[group_col] = df_res
    return dict_df_res


# ----------------------------------------------------------------------
# Correlation analysis
# ----------------------------------------------------------------------
def downstream_corr_analysis(
    adata, df_full_score: pd.DataFrame, var_cols: List[str]
) -> pd.DataFrame:
    """scDRS cell-level correlation analysis.

    For each continuous variable in ``var_cols``, test the
    disease-variable association via a control-score-based Monte-Carlo
    test on Pearson's correlation.

    Parameters
    ----------
    adata
        AnnData; ``var_cols`` must be columns of ``adata.obs``.
    df_full_score
        ``score_cell`` output with ``return_ctrl_norm_score=True``.
    var_cols
        Continuous cell-level variables in ``adata.obs``.

    Returns
    -------
    pandas.DataFrame
        Per-variable ``n_ctrl, corr_mcp, corr_mcz``.
    """
    assert len(set(var_cols) - set(adata.obs)) == 0, (
        "Missing `var_cols` variables from `adata.obs.columns`."
    )
    cell_list = sorted(set(adata.obs_names) & set(df_full_score.index))
    control_list = [x for x in df_full_score.columns if x.startswith("ctrl_norm_score")]
    n_ctrl = len(control_list)
    df_reg = adata.obs.loc[cell_list, var_cols].copy()
    df_reg = df_reg.join(df_full_score.loc[cell_list, ["norm_score"] + control_list])

    col_list = ["n_ctrl", "corr_mcp", "corr_mcz"]
    df_res = pd.DataFrame(index=var_cols, columns=col_list, dtype=np.float32)
    for var_col in var_cols:
        corr_ = np.corrcoef(df_reg[var_col], df_reg["norm_score"])[0, 1]
        v_corr_ = np.array(
            [np.corrcoef(df_reg[var_col], df_reg[x])[0, 1] for x in control_list]
        )
        mc_p = ((v_corr_ >= corr_).sum() + 1) / (v_corr_.shape[0] + 1)
        mc_z = (corr_ - v_corr_.mean()) / v_corr_.std()
        df_res.loc[var_col] = [n_ctrl, mc_p, mc_z]
    return df_res


# ----------------------------------------------------------------------
# Gene-level analysis
# ----------------------------------------------------------------------
def downstream_gene_analysis(adata, df_full_score: pd.DataFrame) -> pd.DataFrame:
    """scDRS gene-level correlation analysis.

    Correlate every gene's expression with the disease ``norm_score``.

    Parameters
    ----------
    adata
        AnnData (n_cell, n_gene).
    df_full_score
        ``score_cell`` output containing ``norm_score``.

    Returns
    -------
    pandas.DataFrame
        Per-gene ``CORR`` (Pearson) and ``RANK``, sorted by ``CORR``.
    """
    cell_list = sorted(set(adata.obs_names) & set(df_full_score.index))
    df_reg = df_full_score.loc[cell_list, ["norm_score"]]
    mat_expr = adata[cell_list].X.copy()
    v_corr = _pearson_corr(mat_expr, df_reg["norm_score"].values)
    df_res = pd.DataFrame(
        index=adata.var_names, columns=["CORR", "RANK"], dtype=np.float32
    )
    df_res["CORR"] = v_corr
    df_res.sort_values("CORR", ascending=False, inplace=True)
    df_res["RANK"] = np.arange(df_res.shape[0])
    return df_res


# ----------------------------------------------------------------------
# Geary's C subroutines
# ----------------------------------------------------------------------
def test_gearysc(
    adata, df_full_score: pd.DataFrame, groupby: str,
    opt: str = "control_distribution_match",
) -> pd.DataFrame:
    """Significance of Geary's C heterogeneity per cell group.

    Parameters
    ----------
    adata
        AnnData with a ``connectivities`` graph; obs order must match
        ``df_full_score.index``.
    df_full_score
        DataFrame with ``norm_score`` and ``ctrl_norm_score_*`` columns.
    groupby
        Grouping column.
    opt
        Null construction; default ``control_distribution_match``.

    Returns
    -------
    pandas.DataFrame
        Per-group ``pval, trait, ctrl_mean, ctrl_std, zsc``.
    """
    assert np.all(df_full_score.index == adata.obs_names), (
        "adata.obs_names must match df_full_score.index"
    )
    norm_score = df_full_score["norm_score"]
    ctrl_norm_score = df_full_score[
        [c for c in df_full_score.columns if c.startswith("ctrl_norm_score_")]
    ]
    n_null = ctrl_norm_score.shape[1]
    df_meta = adata.obs.copy()
    df_stats = pd.DataFrame(
        index=df_meta[groupby].unique(),
        columns=["trait"] + [f"null_{i}" for i in range(n_null)],
        data=np.nan,
    )

    def distribution_match(v, ref):
        return np.sort(ref)[rankdata(v, method="ordinal") - 1]

    for group, df_group in df_meta.groupby(groupby):
        group_index = df_group.index
        group_adata = adata[group_index]
        group_norm_score = norm_score[group_index]
        group_ctrl_norm_score = ctrl_norm_score.loc[group_index, :]

        if opt == "control_distribution_match":
            df_stats.loc[group, "trait"] = gearys_c(
                group_adata, group_norm_score.values
            )
            for i_null in range(n_null):
                df_stats.loc[group, f"null_{i_null}"] = gearys_c(
                    group_adata,
                    distribution_match(
                        group_ctrl_norm_score.iloc[:, i_null].values,
                        ref=group_norm_score,
                    ),
                )
        elif opt == "permutation":
            df_stats.loc[group, "trait"] = gearys_c(
                group_adata, group_norm_score.values
            )
            for i_null in range(n_null):
                df_stats.loc[group, f"null_{i_null}"] = gearys_c(
                    group_adata, np.random.permutation(group_norm_score.values)
                )
        elif opt == "control":
            df_stats.loc[group, "trait"] = gearys_c(
                group_adata, group_norm_score.values
            )
            for i_null in range(n_null):
                df_stats.loc[group, f"null_{i_null}"] = gearys_c(
                    group_adata, group_ctrl_norm_score.iloc[:, i_null].values
                )
        else:
            raise NotImplementedError

    trait_col = "trait"
    ctrl_cols = [c for c in df_stats.columns if c.startswith("null_")]
    pval = (
        (df_stats[trait_col].values > df_stats[ctrl_cols].values.T).sum(axis=0) + 1
    ) / (len(ctrl_cols) + 1)
    pval = pval.astype(float)
    pval[np.isnan(df_stats[trait_col].values.astype(float))] = np.nan

    df_rls = pd.DataFrame(
        {
            "pval": pval,
            "trait": df_stats[trait_col].values,
            "ctrl_mean": df_stats[ctrl_cols].mean(axis=1).values,
            "ctrl_std": df_stats[ctrl_cols].std(axis=1).values,
        },
        index=df_stats.index,
    )
    df_rls["zsc"] = (
        -(df_rls[trait_col].values - df_rls["ctrl_mean"]) / df_rls["ctrl_std"]
    )
    return df_rls


def gearys_c(adata, vals) -> float:
    """Geary's C spatial-autocorrelation statistic.

    ``C = (N-1) * sum_ij w_ij (x_i - x_j)^2 / (2 W sum_i (x_i - xbar)^2)``
    using the ``connectivities`` graph in ``adata.obsp``.
    """
    graph = adata.obsp["connectivities"]
    assert graph.shape[0] == graph.shape[1]
    graph_data = graph.data.astype(np.float64, copy=False)
    vals = np.asarray(vals)
    assert graph.shape[0] == vals.shape[0]
    assert np.ndim(vals) == 1

    W = graph_data.sum()
    N = len(graph.indptr) - 1
    vals_bar = vals.mean()
    vals = vals.astype(np.float64)

    total = 0.0
    for i in range(N):
        s = slice(graph.indptr[i], graph.indptr[i + 1])
        i_indices = graph.indices[s]
        i_data = graph_data[s]
        total += np.sum(i_data * ((vals[i] - vals[i_indices]) ** 2))

    numer = (N - 1) * total
    denom = 2 * W * ((vals - vals_bar) ** 2).sum()
    return numer / denom


# ----------------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------------
def _pearson_corr(mat_X, mat_Y):
    """Pearson correlation between columns of ``mat_X`` and ``mat_Y``."""
    if sp.sparse.issparse(mat_X) | sp.sparse.issparse(mat_Y):
        return _pearson_corr_sparse(mat_X, mat_Y)
    if np.ndim(mat_X) == 1:
        mat_X = mat_X.reshape([-1, 1])
    if np.ndim(mat_Y) == 1:
        mat_Y = mat_Y.reshape([-1, 1])
    mat_X = (mat_X - mat_X.mean(axis=0)) / mat_X.std(axis=0).clip(min=1e-8)
    mat_Y = (mat_Y - mat_Y.mean(axis=0)) / mat_Y.std(axis=0).clip(min=1e-8)
    mat_corr = np.array(mat_X.T.dot(mat_Y) / mat_X.shape[0], dtype=np.float32)
    if (mat_X.shape[1] == 1) | (mat_Y.shape[1] == 1):
        return mat_corr.reshape([-1])
    return mat_corr


def _pearson_corr_sparse(mat_X, mat_Y):
    """Pearson correlation for sparse inputs."""
    from .preprocess import _get_mean_var

    if np.ndim(mat_X) == 1:
        mat_X = mat_X.reshape([-1, 1])
    if np.ndim(mat_Y) == 1:
        mat_Y = mat_Y.reshape([-1, 1])
    if not sp.sparse.issparse(mat_X):
        mat_X = sp.sparse.csr_matrix(mat_X)
    if not sp.sparse.issparse(mat_Y):
        mat_Y = sp.sparse.csr_matrix(mat_Y)

    v_X_mean, v_X_var = _get_mean_var(mat_X, axis=0)
    v_X_sd = np.sqrt(v_X_var).clip(min=1e-8)
    v_Y_mean, v_Y_var = _get_mean_var(mat_Y, axis=0)
    v_Y_sd = np.sqrt(v_Y_var).clip(min=1e-8)

    mat_corr = mat_X.T.dot(mat_Y) / mat_X.shape[0]
    mat_corr = mat_corr - v_X_mean.reshape([-1, 1]).dot(v_Y_mean.reshape([1, -1]))
    mat_corr = mat_corr / v_X_sd.reshape([-1, 1]).dot(v_Y_sd.reshape([1, -1]))
    mat_corr = np.array(mat_corr, dtype=np.float32)
    if (mat_X.shape[1] == 1) | (mat_Y.shape[1] == 1):
        return mat_corr.reshape([-1])
    return mat_corr
