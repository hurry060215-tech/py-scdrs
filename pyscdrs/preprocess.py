"""Preprocessing for scDRS — covariate correction and gene/cell statistics.

This is a clean reimplementation of ``scdrs.pp`` (the preprocessing
module of the original scDRS package).  It is numerically faithful to the
original: covariate regression, gene-level mean/variance, the loess
technical-variance fit (``scikit-misc`` loess, span=0.3, degree=2) and the
``n_mean_bin * n_var_bin`` mean-variance binning all match.

The preprocessing result is stored in ``adata.uns["SCDRS_PARAM"]`` and is
required input for :func:`pyscdrs.score.score_cell`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "preprocess",
    "compute_stats",
    "reg_out",
    "category2dummy",
]


# ----------------------------------------------------------------------
# Covariate dataframe helper
# ----------------------------------------------------------------------
def category2dummy(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert categorical columns of a dataframe to binary dummy variables.

    Parameters
    ----------
    df
        Dataframe to convert.
    cols
        Columns to convert.  If ``None``, non-numeric columns are auto-detected.
    verbose
        Print which columns were converted.

    Returns
    -------
    pandas.DataFrame
        Dataframe with categorical columns replaced by ``drop_first`` dummies.
    """
    df = df.copy()

    if cols is None:
        cols = list(set(df.columns) - set(df._get_numeric_data().columns))

    assert set(cols).issubset(set(df.columns)), "'cols' must be a subset of df.columns"

    cols_to_drop: List[str] = []
    cols_to_add: List[str] = []
    dummy_dfs: List[pd.DataFrame] = []
    for col in cols:
        dummy_df = pd.get_dummies(df[col], drop_first=True)
        dummy_df.columns = [f"{col}_{s}" for s in dummy_df.columns]
        dummy_df.loc[df[col].isnull(), dummy_df.columns] = np.nan
        cols_to_drop.append(col)
        cols_to_add.extend(dummy_df.columns)
        dummy_dfs.append(dummy_df)

    df = pd.concat([df, *dummy_dfs], axis=1).drop(columns=cols_to_drop)

    if (len(cols_to_add) > 0) and verbose:
        print(
            "pyscdrs.preprocess.category2dummy: "
            f"detected categorical columns: {','.join(cols)}; "
            f"added dummy columns: {','.join(cols_to_add)}; "
            f"dropped columns: {','.join(cols_to_drop)}."
        )
    return df


# ----------------------------------------------------------------------
# Linear-algebra subroutines
# ----------------------------------------------------------------------
def reg_out(mat_Y, mat_X):
    """Regress ``mat_X`` out of ``mat_Y`` (ordinary least squares residual).

    Parameters
    ----------
    mat_Y
        Response variable of shape ``(n_sample, n_response)``.
    mat_X
        Covariates of shape ``(n_sample, n_covariate)``.

    Returns
    -------
    numpy.ndarray
        Residual ``mat_Y - mat_X @ beta``.
    """
    if sparse.issparse(mat_X):
        mat_X = mat_X.toarray()
    else:
        mat_X = np.array(mat_X)
    if mat_X.ndim == 1:
        mat_X = mat_X.reshape([-1, 1])

    if sparse.issparse(mat_Y):
        mat_Y = mat_Y.toarray()
    else:
        mat_Y = np.array(mat_Y)
    if mat_Y.ndim == 1:
        mat_Y = mat_Y.reshape([-1, 1])

    n_sample = mat_Y.shape[0]
    mat_xtx = np.dot(mat_X.T, mat_X) / n_sample
    mat_xty = np.dot(mat_X.T, mat_Y) / n_sample
    mat_coef = np.linalg.solve(mat_xtx, mat_xty)
    mat_Y_resid = mat_Y - mat_X.dot(mat_coef)

    if mat_Y_resid.shape[1] == 1:
        mat_Y_resid = mat_Y_resid.reshape([-1])
    return mat_Y_resid


def _get_mean_var(mat_X, axis: int = 0, weights=None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance of a dense or sparse matrix along ``axis``.

    Variance uses the population (``ddof=0``) estimator, matching scDRS.
    """
    if weights is None:
        if sparse.issparse(mat_X):
            v_mean = np.asarray(mat_X.mean(axis=axis)).reshape([-1])
            v_var = np.asarray(mat_X.power(2).mean(axis=axis)).reshape([-1])
            v_var = v_var - v_mean ** 2
        else:
            mat_X = np.asarray(mat_X)
            v_mean = np.mean(mat_X, axis=axis)
            v_var = np.var(mat_X, axis=axis)
    else:
        weights = np.asarray(weights)
        if sparse.issparse(mat_X):
            v_mean = _weighted_sparse_average(mat_X, weights, axis=axis)
            v_var = _weighted_sparse_average(mat_X.power(2), weights, axis=axis)
            v_var = v_var - v_mean ** 2
        else:
            mat_X = np.asarray(mat_X)
            v_mean = np.average(mat_X, axis=axis, weights=weights)
            v_var = np.average(np.square(mat_X), axis=axis, weights=weights)
            v_var = v_var - v_mean ** 2
    return v_mean, v_var


def _weighted_sparse_average(mat_X, weights, axis: int = 0) -> np.ndarray:
    """Weighted mean of a sparse matrix along ``axis``."""
    if axis == 0:
        v_mean = mat_X.T.multiply(weights).mean(axis=1)
        return np.asarray(v_mean).reshape([-1]) / np.mean(weights)
    if axis == 1:
        v_mean = mat_X.multiply(weights).mean(axis=1)
        return np.asarray(v_mean).reshape([-1]) / np.mean(weights)
    raise ValueError("axis must be 0 or 1")


def _get_mean_var_implicit_cov_corr(
    adata, axis: int = 0, weights=None, transform_func=None, n_chunk: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean / variance of the implicitly covariate-corrected matrix.

    Computes statistics of ``adata.X + COV_MAT @ COV_BETA + COV_GENE_MEAN``
    iteratively in chunks, so that the original sparse ``adata.X`` is never
    densified all at once.
    """
    assert axis in (0, 1), "axis must be one of [0, 1]"
    assert "SCDRS_PARAM" in adata.uns, (
        "adata.uns['SCDRS_PARAM'] not found, run `preprocess` first"
    )

    n_obs, n_gene = adata.shape
    cell_list = list(adata.obs_names)
    cov_list = list(adata.uns["SCDRS_PARAM"]["COV_MAT"])
    gene_list = list(adata.var_names)
    cov_mat = adata.uns["SCDRS_PARAM"]["COV_MAT"].loc[cell_list, cov_list].values
    cov_beta = adata.uns["SCDRS_PARAM"]["COV_BETA"].loc[gene_list, cov_list].values.T
    gene_mean = adata.uns["SCDRS_PARAM"]["COV_GENE_MEAN"].loc[gene_list].values

    if transform_func is None:
        transform_func = lambda x: x
    if weights is not None:
        weights = np.asarray(weights)

    if axis == 0:
        v_mean = np.zeros(n_gene)
        v_var = np.zeros(n_gene)
        start = 0
        chunk_size = n_gene // n_chunk
        while start < n_gene:
            stop = min(start + chunk_size, n_gene)
            chunk_X = (
                adata.X[:, start:stop]
                + cov_mat @ cov_beta[:, start:stop]
                + gene_mean[start:stop]
            )
            chunk_X = transform_func(chunk_X)
            v_mean[start:stop], v_var[start:stop] = _get_mean_var(
                chunk_X, axis=axis, weights=weights
            )
            start = stop
    else:
        v_mean = np.zeros(n_obs)
        v_var = np.zeros(n_obs)
        start = 0
        chunk_size = n_obs // n_chunk
        while start < n_obs:
            stop = min(start + chunk_size, n_obs)
            chunk_X = adata.X[start:stop, :] + cov_mat[start:stop] @ cov_beta + gene_mean
            chunk_X = transform_func(chunk_X)
            v_mean[start:stop], v_var[start:stop] = _get_mean_var(
                chunk_X, axis=axis, weights=weights
            )
            start = stop
    return v_mean, v_var


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def compute_stats(
    adata,
    implicit_cov_corr: bool = False,
    cell_weight=None,
    n_mean_bin: int = 20,
    n_var_bin: int = 20,
    n_chunk: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute gene-level and cell-level statistics for scDRS scoring.

    Parameters
    ----------
    adata
        Single-cell data (n_cell, n_gene), assumed size-factor normalized
        and log1p transformed.
    implicit_cov_corr
        If ``True``, statistics are computed for the implicitly
        covariate-corrected data using ``adata.uns["SCDRS_PARAM"]``.
    cell_weight
        Optional per-cell weights (length n_cell).
    n_mean_bin, n_var_bin
        Number of mean / variance bins for the mean-variance grid.
    n_chunk
        Number of chunks for the implicit covariate-correction path.

    Returns
    -------
    df_gene : pandas.DataFrame
        Gene statistics with columns ``mean, var, var_tech, ct_mean,
        ct_var, ct_var_tech, mean_var``.
    df_cell : pandas.DataFrame
        Cell statistics with columns ``mean, var``.
    """
    from skmisc.loess import loess

    if implicit_cov_corr:
        assert "SCDRS_PARAM" in adata.uns, (
            "adata.uns['SCDRS_PARAM'] not found, run `preprocess` first"
        )

    df_gene = pd.DataFrame(
        index=adata.var_names,
        columns=[
            "mean",
            "var",
            "var_tech",
            "ct_mean",
            "ct_var",
            "ct_var_tech",
            "mean_var",
        ],
    )
    df_cell = pd.DataFrame(index=adata.obs_names, columns=["mean", "var"])

    # Gene-level statistics
    if not implicit_cov_corr:
        df_gene["mean"], df_gene["var"] = _get_mean_var(
            adata.X, axis=0, weights=cell_weight
        )
        if sparse.issparse(adata.X):
            temp_X = adata.X.copy().expm1()
        else:
            temp_X = np.expm1(adata.X)
        df_gene["ct_mean"], df_gene["ct_var"] = _get_mean_var(
            temp_X, axis=0, weights=cell_weight
        )
        del temp_X
    else:
        df_gene["mean"], df_gene["var"] = _get_mean_var_implicit_cov_corr(
            adata, axis=0, n_chunk=n_chunk, weights=cell_weight
        )
        df_gene["ct_mean"], df_gene["ct_var"] = _get_mean_var_implicit_cov_corr(
            adata, transform_func=np.expm1, axis=0, n_chunk=n_chunk, weights=cell_weight
        )

    # Technical variance via loess fit (Seurat v3 / Frost NAR 2020 recipe)
    not_const = df_gene["ct_var"].values > 0
    estimat_var = np.zeros(adata.shape[1], dtype=np.float64)
    y = np.log10(df_gene["ct_var"].values[not_const])
    x = np.log10(df_gene["ct_mean"].values[not_const])
    model = loess(x, y, span=0.3, degree=2)
    model.fit()
    estimat_var[not_const] = model.outputs.fitted_values
    df_gene["ct_var_tech"] = 10 ** estimat_var
    df_gene["var_tech"] = df_gene["var"] * df_gene["ct_var_tech"] / df_gene["ct_var"]
    df_gene.loc[df_gene["var_tech"].isna(), "var_tech"] = 0

    # Mean-variance bins
    n_bin_max = np.floor(np.sqrt(adata.shape[1] / 10)).astype(int)
    if (n_mean_bin > n_bin_max) | (n_var_bin > n_bin_max):
        n_mean_bin, n_var_bin = n_bin_max, n_bin_max
        print(
            "Too few genes for 20*20 bins, setting n_mean_bin=n_var_bin=%d" % n_bin_max
        )
    v_mean_bin = pd.qcut(df_gene["mean"], n_mean_bin, labels=False, duplicates="drop")
    df_gene["mean_var"] = ""
    for bin_ in sorted(set(v_mean_bin)):
        ind_select = v_mean_bin == bin_
        v_var_bin = pd.qcut(
            df_gene.loc[ind_select, "var"], n_var_bin, labels=False, duplicates="drop"
        )
        df_gene.loc[ind_select, "mean_var"] = ["%d.%d" % (bin_, x) for x in v_var_bin]

    # Cell-level statistics
    if not implicit_cov_corr:
        df_cell["mean"], df_cell["var"] = _get_mean_var(adata.X, axis=1)
    else:
        df_cell["mean"], df_cell["var"] = _get_mean_var_implicit_cov_corr(
            adata, axis=1, n_chunk=n_chunk
        )

    return df_gene, df_cell


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def preprocess(
    data,
    cov: Optional[pd.DataFrame] = None,
    adj_prop: Optional[str] = None,
    n_mean_bin: int = 20,
    n_var_bin: int = 20,
    n_chunk: Optional[int] = None,
    copy: bool = False,
):
    """Preprocess single-cell data for scDRS analysis.

    Faithful reimplementation of ``scdrs.pp.preprocess``.  It

    1. regresses out covariates (adding back the per-gene mean), and
    2. computes gene-/cell-level statistics and the mean-variance bins.

    Results are stored in ``data.uns["SCDRS_PARAM"]``.  When ``data.X`` is
    sparse and ``cov`` is given, the implicit-covariate-correction mode is
    used: ``data.X`` is left untouched and the correction is applied
    on-the-fly during statistic computation and scoring.

    Parameters
    ----------
    data
        AnnData of shape (n_cell, n_gene), size-factor normalized and
        log1p transformed.
    cov
        Covariate dataframe (n_cell, n_cov).  Should contain a constant
        term and cover at least 75% of cells.
    adj_prop
        Cell-group column in ``data.obs`` used to inverse-weight cells by
        group size when computing gene statistics.
    n_mean_bin, n_var_bin
        Number of mean / variance bins for control-gene matching.
    n_chunk
        Number of chunks for the implicit covariate-correction path.
    copy
        Return a modified copy instead of writing to ``data`` in place.

    Returns
    -------
    anndata.AnnData or None
        Modified copy when ``copy=True``, otherwise ``None`` (in place).
    """
    adata = data.copy() if copy else data
    n_cell, n_gene = adata.shape

    flag_sparse = sparse.issparse(adata.X)
    flag_cov = cov is not None
    flag_adj_prop = adj_prop is not None
    adata.uns["SCDRS_PARAM"] = {
        "FLAG_SPARSE": flag_sparse,
        "FLAG_COV": flag_cov,
        "FLAG_ADJ_PROP": flag_adj_prop,
    }

    # Standardize adata.X type
    if flag_sparse:
        if not isinstance(adata.X, sparse.csr_matrix):
            adata.X = sparse.csr_matrix(adata.X)
    else:
        if not isinstance(adata.X, np.ndarray):
            adata.X = np.array(adata.X)

    # Covariate correction
    if flag_cov:
        assert len(set(cov.index) & set(adata.obs_names)) > 0.75 * n_cell, (
            "cov does not match the cells in data"
        )

        df_cov = pd.DataFrame(index=adata.obs_names)
        df_cov = df_cov.join(cov)
        df_cov = category2dummy(df=df_cov, verbose=True)
        df_cov.fillna(df_cov.mean(), inplace=True)

        # Add a constant term if not already (approximately) present
        v_resid = reg_out(np.ones(n_cell), df_cov.values)
        if (v_resid ** 2).mean() > 0.01:
            df_cov["SCDRS_CONST"] = 1

        v_gene_mean = np.array(adata.X.mean(axis=0)).flatten()
        if flag_sparse:
            mat_beta = np.linalg.solve(
                np.dot(df_cov.values.T, df_cov.values) / n_cell,
                sparse.csr_matrix.dot(df_cov.values.T, adata.X) / n_cell,
            )
            adata.uns["SCDRS_PARAM"]["COV_MAT"] = df_cov
            adata.uns["SCDRS_PARAM"]["COV_BETA"] = pd.DataFrame(
                -mat_beta.T, index=adata.var_names, columns=df_cov.columns
            )
            adata.uns["SCDRS_PARAM"]["COV_GENE_MEAN"] = pd.Series(
                v_gene_mean, index=adata.var_names
            )
        else:
            adata.X = reg_out(adata.X, df_cov.values)
            adata.X += v_gene_mean

    # Mode and chunk setting
    if flag_sparse and flag_cov:
        implicit_cov_corr = True
        if n_chunk is None:
            n_chunk = 5 * adata.shape[0] * adata.shape[1] // adata.X.data.shape[0] + 1
    else:
        implicit_cov_corr = False
        if n_chunk is None:
            n_chunk = 20

    # Cell-group proportion adjustment
    if flag_adj_prop:
        assert adj_prop in adata.obs, (
            "'adj_prop'=%s not in 'adata.obs.columns'" % adj_prop
        )
        assert adata.obs[adj_prop].unique().shape[0] < 0.1 * n_cell, (
            "On average <10 cells per group, maybe `%s` is not categorical?" % adj_prop
        )
        temp_df = adata.obs[[adj_prop]].copy()
        temp_df["cell"] = 1
        temp_df = temp_df.groupby(adj_prop).agg({"cell": len})
        temp_df["cell"] = temp_df["cell"].clip(lower=int(0.01 * temp_df["cell"].max()))
        temp_dic = {x: n_cell / temp_df.loc[x, "cell"] for x in temp_df.index}
        cell_weight = np.array([temp_dic[x] for x in adata.obs[adj_prop]])
        cell_weight = cell_weight / cell_weight.mean()
    else:
        cell_weight = None

    df_gene, df_cell = compute_stats(
        adata,
        implicit_cov_corr=implicit_cov_corr,
        cell_weight=cell_weight,
        n_mean_bin=n_mean_bin,
        n_var_bin=n_var_bin,
        n_chunk=n_chunk,
    )

    adata.uns["SCDRS_PARAM"]["GENE_STATS"] = df_gene
    adata.uns["SCDRS_PARAM"]["CELL_STATS"] = df_cell

    return adata if copy else None
