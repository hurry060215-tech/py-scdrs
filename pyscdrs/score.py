"""Core scDRS disease-relevance scoring.

Clean reimplementation of ``scdrs.method.score_cell`` and its
subroutines.  The algorithm:

1. Compute a raw disease score per cell as a weighted average of
   covariate-corrected expression over the disease gene set.
2. Draw ``n_ctrl`` control gene sets matched to the disease genes on a
   gene-level statistic (mean-variance bins by default).
3. Background-correct (two gene-set alignments + cell-wise
   standardization) to obtain normalized scores.
4. Derive per-cell Monte-Carlo p-values and a pooled empirical p-value.

With the same ``random_seed`` the control gene sets, and therefore the
p-values, match the original scDRS exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse

__all__ = ["score_cell"]

_WEIGHT_OPTS = {"uniform", "vs", "inv_std", "od"}


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def score_cell(
    data,
    gene_list,
    gene_weight=None,
    ctrl_match_key: str = "mean_var",
    n_ctrl: int = 1000,
    n_genebin: int = 200,
    weight_opt: str = "vs",
    copy: bool = False,
    return_ctrl_raw_score: bool = False,
    return_ctrl_norm_score: bool = False,
    random_seed: int = 0,
    verbose: bool = False,
    save_intermediate: Optional[str] = None,
) -> pd.DataFrame:
    """Score cells for disease relevance against a disease gene set.

    Requires ``data.uns["SCDRS_PARAM"]`` (run :func:`pyscdrs.preprocess`
    first).

    Parameters
    ----------
    data
        AnnData (n_cell, n_gene), size-factor normalized and log1p
        transformed.
    gene_list
        Disease gene list.
    gene_weight
        Optional per-gene weights (length ``len(gene_list)``).  Defaults
        to all-ones.
    ctrl_match_key
        Gene-level statistic for matching control genes; must be a column
        of ``data.uns["SCDRS_PARAM"]["GENE_STATS"]`` (default ``mean_var``).
    n_ctrl
        Number of control gene sets (Monte-Carlo replicates).
    n_genebin
        Number of bins when ``ctrl_match_key`` is treated as continuous.
    weight_opt
        Raw-score weighting: ``uniform``, ``vs`` (1/sqrt of technical
        variance), ``inv_std`` (1/std) or ``od`` (overdispersion score).
    copy
        Operate on a copy of ``data``.
    return_ctrl_raw_score, return_ctrl_norm_score
        Include per-control raw / normalized scores in the output.
    random_seed
        Random seed governing the control gene sets.
    verbose
        Print a run summary.
    save_intermediate
        File-path prefix for dumping intermediate background-correction
        matrices.

    Returns
    -------
    pandas.DataFrame
        Per-cell results (``dtype=float32``) with columns
        ``raw_score, norm_score, mc_pval, pval, nlog10_pval, zscore`` and
        optionally ``ctrl_raw_score_*`` / ``ctrl_norm_score_*``.
    """
    np.random.seed(random_seed)
    adata = data.copy() if copy else data
    n_cell, n_gene = adata.shape

    assert "SCDRS_PARAM" in adata.uns, (
        "adata.uns['SCDRS_PARAM'] not found, run `pyscdrs.preprocess` first"
    )
    assert "GENE_STATS" in adata.uns["SCDRS_PARAM"], (
        "adata.uns['SCDRS_PARAM']['GENE_STATS'] not found, run `pyscdrs.preprocess` first"
    )
    gene_stats_set_expect = {"mean", "var", "var_tech"}
    gene_stats_set = set(adata.uns["SCDRS_PARAM"]["GENE_STATS"])
    assert len(gene_stats_set_expect - gene_stats_set) == 0, (
        "One of 'mean', 'var', 'var_tech' not in GENE_STATS; run `preprocess` first"
    )
    assert ctrl_match_key in adata.uns["SCDRS_PARAM"]["GENE_STATS"], (
        "ctrl_match_key=%s not found in GENE_STATS" % ctrl_match_key
    )
    assert weight_opt in _WEIGHT_OPTS, (
        "weight_opt=%s is not one of %s" % (weight_opt, _WEIGHT_OPTS)
    )

    if verbose:
        msg = "# pyscdrs.score_cell summary:"
        msg += "\n    n_cell=%d, n_gene=%d," % (n_cell, n_gene)
        msg += "\n    n_disease_gene=%d," % len(gene_list)
        msg += "\n    n_ctrl=%d, n_genebin=%d," % (n_ctrl, n_genebin)
        msg += "\n    ctrl_match_key='%s'," % ctrl_match_key
        msg += "\n    weight_opt='%s'," % weight_opt
        msg += "\n    random_seed=%d." % random_seed
        print(msg)

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[adata.var_names].copy()
    df_gene["gene"] = df_gene.index
    df_gene.drop_duplicates(subset="gene", inplace=True)

    gene_list = list(gene_list)
    if gene_weight is not None:
        gene_weight = list(gene_weight)
    else:
        gene_weight = [1] * len(gene_list)

    # Restrict to overlapping genes, sorted for reproducibility
    dic_gene_weight = {x: y for x, y in zip(gene_list, gene_weight)}
    gene_list = sorted(set(gene_list) & set(df_gene["gene"]))
    gene_weight = [dic_gene_weight[x] for x in gene_list]

    if verbose:
        print("# pyscdrs.score_cell: use %d overlapping genes" % len(gene_list))

    # Control gene sets
    dic_ctrl_list, dic_ctrl_weight = _select_ctrl_geneset(
        df_gene, gene_list, gene_weight, ctrl_match_key, n_ctrl, n_genebin, random_seed
    )

    # Raw scores
    v_raw_score, v_score_weight = _compute_raw_score(
        adata, gene_list, gene_weight, weight_opt
    )

    mat_ctrl_raw_score = np.zeros([n_cell, n_ctrl])
    mat_ctrl_weight = np.zeros([len(gene_list), n_ctrl])
    ctrl_iter = range(n_ctrl)
    if verbose:
        try:
            from tqdm import tqdm

            ctrl_iter = tqdm(ctrl_iter, desc="Computing control scores")
        except ImportError:
            pass
    for i_ctrl in ctrl_iter:
        v_ctrl_raw_score, v_ctrl_weight = _compute_raw_score(
            adata, dic_ctrl_list[i_ctrl], dic_ctrl_weight[i_ctrl], weight_opt
        )
        mat_ctrl_raw_score[:, i_ctrl] = v_ctrl_raw_score
        mat_ctrl_weight[:, i_ctrl] = v_ctrl_weight

    # Control-to-trait variance ratio (only for weighted-average scores)
    v_var_ratio_c2t = np.ones(n_ctrl)
    if (ctrl_match_key == "mean_var") & (weight_opt in ["uniform", "vs", "inv_std"]):
        for i_ctrl in range(n_ctrl):
            v_var_ratio_c2t[i_ctrl] = (
                df_gene.loc[dic_ctrl_list[i_ctrl], "var"]
                * mat_ctrl_weight[:, i_ctrl] ** 2
            ).sum()
        v_var_ratio_c2t /= (df_gene.loc[gene_list, "var"] * v_score_weight ** 2).sum()

    v_norm_score, mat_ctrl_norm_score = _correct_background(
        v_raw_score, mat_ctrl_raw_score, v_var_ratio_c2t, save_intermediate
    )

    # P-values
    mc_p = (1 + (mat_ctrl_norm_score.T >= v_norm_score).sum(axis=0)) / (1 + n_ctrl)
    pooled_p = _get_p_from_empi_null(v_norm_score, mat_ctrl_norm_score.flatten())
    nlog10_pooled_p = -np.log10(pooled_p)
    pooled_z = -sp.stats.norm.ppf(pooled_p).clip(min=-10, max=10)

    dic_res = {
        "raw_score": v_raw_score,
        "norm_score": v_norm_score,
        "mc_pval": mc_p,
        "pval": pooled_p,
        "nlog10_pval": nlog10_pooled_p,
        "zscore": pooled_z,
    }
    if return_ctrl_raw_score:
        for i in range(n_ctrl):
            dic_res["ctrl_raw_score_%d" % i] = mat_ctrl_raw_score[:, i]
    if return_ctrl_norm_score:
        for i in range(n_ctrl):
            dic_res["ctrl_norm_score_%d" % i] = mat_ctrl_norm_score[:, i]
    df_res = pd.DataFrame(index=adata.obs.index, data=dic_res, dtype=np.float32)
    return df_res


# ----------------------------------------------------------------------
# Control gene set selection
# ----------------------------------------------------------------------
def _select_ctrl_geneset(
    input_df_gene: pd.DataFrame,
    gene_list: List[str],
    gene_weight: List[float],
    ctrl_match_key: str,
    n_ctrl: int,
    n_genebin: int,
    random_seed: int,
) -> Tuple[Dict[int, list], Dict[int, list]]:
    """Select ``n_ctrl`` control gene sets matched on ``ctrl_match_key``.

    A categorical ``ctrl_match_key`` (fewer than 10% unique values) is
    matched within each category; a continuous one is binned into
    ``n_genebin`` quantile bins.  Control genes inherit the weight of the
    disease gene they replace.
    """
    np.random.seed(random_seed)
    df_gene = input_df_gene.copy()
    if "gene" not in df_gene:
        df_gene["gene"] = df_gene.index
    disease_gene_set = set(gene_list)
    dic_gene_weight = {x: y for x, y in zip(gene_list, gene_weight)}

    if df_gene[ctrl_match_key].unique().shape[0] < df_gene.shape[0] / 10:
        df_gene_bin = df_gene.groupby(ctrl_match_key).agg({"gene": set})
    else:
        df_gene["qbin"] = pd.qcut(
            df_gene[ctrl_match_key], q=n_genebin, labels=False, duplicates="drop"
        )
        df_gene_bin = df_gene.groupby("qbin").agg({"gene": set})

    dic_ctrl_list: Dict[int, list] = {x: [] for x in range(n_ctrl)}
    dic_ctrl_weight: Dict[int, list] = {x: [] for x in range(n_ctrl)}
    for bin_ in df_gene_bin.index:
        bin_gene = sorted(df_gene_bin.loc[bin_, "gene"])
        bin_disease_gene = sorted(df_gene_bin.loc[bin_, "gene"] & disease_gene_set)
        if len(bin_disease_gene) > 0:
            for i_list in np.arange(n_ctrl):
                dic_ctrl_list[i_list].extend(
                    np.random.choice(
                        bin_gene, size=len(bin_disease_gene), replace=False
                    )
                )
                dic_ctrl_weight[i_list].extend(
                    [dic_gene_weight[x] for x in bin_disease_gene]
                )
    return dic_ctrl_list, dic_ctrl_weight


# ----------------------------------------------------------------------
# Raw score computation
# ----------------------------------------------------------------------
def _compute_raw_score(
    adata, gene_list, gene_weight, weight_opt: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the per-cell raw score for a gene set."""
    gene_list = list(gene_list)
    gene_weight = list(gene_weight)

    assert weight_opt in _WEIGHT_OPTS, (
        "weight_opt=%s is not one of %s" % (weight_opt, _WEIGHT_OPTS)
    )

    if weight_opt == "od":
        return _compute_overdispersion_score(adata, gene_list, gene_weight)

    assert "SCDRS_PARAM" in adata.uns, (
        "adata.uns['SCDRS_PARAM'] not found, run `pyscdrs.preprocess` first"
    )
    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"]
    flag_sparse = adata.uns["SCDRS_PARAM"]["FLAG_SPARSE"]
    flag_cov = adata.uns["SCDRS_PARAM"]["FLAG_COV"]

    if weight_opt == "uniform":
        v_score_weight = np.ones(len(gene_list))
    elif weight_opt == "vs":
        v_score_weight = 1 / np.sqrt(df_gene.loc[gene_list, "var_tech"].values + 1e-2)
    elif weight_opt == "inv_std":
        v_score_weight = 1 / np.sqrt(df_gene.loc[gene_list, "var"].values + 1e-2)

    if gene_weight is not None:
        v_score_weight = v_score_weight * np.array(gene_weight)
    v_score_weight = v_score_weight / v_score_weight.sum()

    if flag_sparse and flag_cov:
        cell_list = list(adata.obs_names)
        cov_list = list(adata.uns["SCDRS_PARAM"]["COV_MAT"])
        cov_mat = adata.uns["SCDRS_PARAM"]["COV_MAT"].loc[cell_list, cov_list].values
        cov_beta = (
            adata.uns["SCDRS_PARAM"]["COV_BETA"].loc[gene_list, cov_list].values.T
        )
        gene_mean = adata.uns["SCDRS_PARAM"]["COV_GENE_MEAN"].loc[gene_list].values
        v_raw_score = (
            adata[:, gene_list].X.dot(v_score_weight)
            + cov_mat @ (cov_beta @ v_score_weight)
            + gene_mean @ v_score_weight
        ).flatten()
    else:
        v_raw_score = np.asarray(
            adata[:, gene_list].X.dot(v_score_weight)
        ).reshape([-1])
    return v_raw_score, v_score_weight


def _compute_overdispersion_score(
    adata, gene_list, gene_weight
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the overdispersion (``od``) raw score."""
    gene_list = list(gene_list)
    gene_weight = list(gene_weight)

    assert "SCDRS_PARAM" in adata.uns, (
        "adata.uns['SCDRS_PARAM'] not found, run `pyscdrs.preprocess` first"
    )
    flag_sparse = adata.uns["SCDRS_PARAM"]["FLAG_SPARSE"]
    flag_cov = adata.uns["SCDRS_PARAM"]["FLAG_COV"]
    if flag_sparse and flag_cov:
        cell_list = list(adata.obs_names)
        cov_list = list(adata.uns["SCDRS_PARAM"]["COV_MAT"])
        mat_X = (
            adata[:, gene_list].X.toarray()
            + adata.uns["SCDRS_PARAM"]["COV_MAT"]
            .loc[cell_list, cov_list]
            .values.dot(
                adata.uns["SCDRS_PARAM"]["COV_BETA"].loc[gene_list, cov_list].values.T
            )
            + adata.uns["SCDRS_PARAM"]["COV_GENE_MEAN"].loc[gene_list].values
        )
    else:
        mat_X = adata[:, gene_list].X

    v_mean = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[gene_list, "mean"].values
    v_var_tech = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[gene_list, "var_tech"].values

    v_w = 1 / (v_var_tech + 1e-2)
    if gene_weight is not None:
        v_w = v_w * np.array(gene_weight)
    v_w = v_w / v_w.sum()

    if sparse.issparse(mat_X):
        v_raw_score = mat_X.power(2).dot(v_w).reshape([-1])
    else:
        v_raw_score = (mat_X ** 2).dot(v_w).reshape([-1])
    v_raw_score = v_raw_score - np.asarray(mat_X.dot(2 * v_w * v_mean)).reshape([-1])
    v_raw_score = v_raw_score + (v_w * (v_mean ** 2 - v_var_tech)).sum()
    return np.asarray(v_raw_score).reshape([-1]), np.ones(len(gene_list))


# ----------------------------------------------------------------------
# Background correction
# ----------------------------------------------------------------------
def _correct_background(
    v_raw_score, mat_ctrl_raw_score, v_var_ratio_c2t, save_intermediate=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-wise and gene-set-wise background correction.

    Two gene-set alignments (mean 0 + matched independent variance) with
    a cell-wise standardization in between, exactly as in scDRS.
    """
    if save_intermediate is not None:
        np.savetxt(
            save_intermediate + ".raw_score.tsv.gz", v_raw_score,
            fmt="%.9e", delimiter="\t",
        )
        np.savetxt(
            save_intermediate + ".ctrl_raw_score.tsv.gz", mat_ctrl_raw_score,
            fmt="%.9e", delimiter="\t",
        )

    ind_zero_score = v_raw_score == 0
    ind_zero_ctrl_score = mat_ctrl_raw_score == 0

    # First gene-set alignment
    v_raw_score = v_raw_score - v_raw_score.mean()
    mat_ctrl_raw_score = mat_ctrl_raw_score - mat_ctrl_raw_score.mean(axis=0)
    mat_ctrl_raw_score = mat_ctrl_raw_score / np.sqrt(v_var_ratio_c2t)

    # Cell-wise standardization
    v_mean = mat_ctrl_raw_score.mean(axis=1)
    v_std = mat_ctrl_raw_score.std(axis=1)
    v_norm_score = v_raw_score.copy()
    v_norm_score = (v_norm_score - v_mean) / v_std
    mat_ctrl_norm_score = ((mat_ctrl_raw_score.T - v_mean) / v_std).T

    # Second gene-set alignment
    v_norm_score = v_norm_score - v_norm_score.mean()
    mat_ctrl_norm_score = mat_ctrl_norm_score - mat_ctrl_norm_score.mean(axis=0)

    # Zero raw scores -> minimum normalized score
    norm_score_min = min(v_norm_score.min(), mat_ctrl_norm_score.min())
    v_norm_score[ind_zero_score] = norm_score_min - 1e-3
    mat_ctrl_norm_score[ind_zero_ctrl_score] = norm_score_min

    if save_intermediate is not None:
        np.savetxt(
            save_intermediate + ".raw_score.final.tsv.gz", v_norm_score,
            fmt="%.9e", delimiter="\t",
        )
        np.savetxt(
            save_intermediate + ".ctrl_raw_score.final.tsv.gz", mat_ctrl_norm_score,
            fmt="%.9e", delimiter="\t",
        )
    return v_norm_score, mat_ctrl_norm_score


def _get_p_from_empi_null(v_t, v_t_null) -> np.ndarray:
    """Empirical p-value of each ``v_t`` element against the null ``v_t_null``.

    ``p = (1 + #{T_i >= T}) / (1 + N)`` computed in O(N log N).
    """
    v_t = np.array(v_t)
    v_t_null = np.sort(np.array(v_t_null))
    v_pos = np.searchsorted(v_t_null, v_t, side="left")
    v_p = (v_t_null.shape[0] - v_pos + 1) / (v_t_null.shape[0] + 1)
    return v_p
