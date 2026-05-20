"""Gene-set and AnnData I/O for scDRS.

Clean reimplementation of the I/O parts of ``scdrs.util`` /
``scdrs.data_loader``:

* :func:`load_gs` — read MAGMA-style ``.gs`` gene-set files.
* :func:`save_gs` — write ``.gs`` files.
* :func:`munge_gs` — build a ``.gs`` file from MAGMA z-score / p-value
  tables.
* :func:`load_h5ad` — read and (optionally) normalize an ``.h5ad`` file.
* :func:`load_homolog_mapping` / :func:`convert_species_name` — mouse /
  human gene-symbol homolog mapping.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "load_gs",
    "save_gs",
    "munge_gs",
    "load_h5ad",
    "load_homolog_mapping",
    "convert_species_name",
    "zsc2pval",
    "pval2zsc",
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------------------
# Species handling
# ----------------------------------------------------------------------
def convert_species_name(species: str) -> str:
    """Normalize a species string to ``mmusculus`` or ``hsapiens``."""
    if species in ["Mouse", "mouse", "Mus_musculus", "mus_musculus", "mmusculus"]:
        return "mmusculus"
    if species in ["Human", "human", "Homo_sapiens", "homo_sapiens", "hsapiens"]:
        return "hsapiens"
    raise ValueError("species name '%s' is not supported" % species)


def load_homolog_mapping(src_species: str, dst_species: str) -> Dict[str, str]:
    """Load a mouse <-> human gene-symbol homolog dictionary.

    Parameters
    ----------
    src_species, dst_species
        Species names (mouse / human aliases); must differ.

    Returns
    -------
    dict
        ``{src_gene_symbol: dst_gene_symbol}``.
    """
    src_species = convert_species_name(src_species)
    dst_species = convert_species_name(dst_species)
    assert src_species != dst_species, "src and dst cannot be the same"

    hom_path = os.path.join(_DATA_DIR, "mouse_human_homologs.txt")
    if not os.path.exists(hom_path):
        raise FileNotFoundError(
            "mouse_human_homologs.txt not bundled with pyscdrs; "
            "homolog conversion is unavailable."
        )
    df_hom = pd.read_csv(hom_path, sep="\t")
    if (src_species == "hsapiens") & (dst_species == "mmusculus"):
        return {x: y for x, y in zip(df_hom["HUMAN_GENE_SYM"], df_hom["MOUSE_GENE_SYM"])}
    if (src_species == "mmusculus") & (dst_species == "hsapiens"):
        return {x: y for x, y in zip(df_hom["MOUSE_GENE_SYM"], df_hom["HUMAN_GENE_SYM"])}
    raise NotImplementedError(
        f"gene conversion from {src_species} to {dst_species} is not supported"
    )


# ----------------------------------------------------------------------
# Gene-set files
# ----------------------------------------------------------------------
def load_gs(
    gs_path: str,
    src_species: Optional[str] = None,
    dst_species: Optional[str] = None,
    to_intersect: Optional[List[str]] = None,
) -> Dict[str, Tuple[List[str], List[float]]]:
    """Load a MAGMA-style ``.gs`` gene-set file.

    The file has two tab-separated columns ``TRAIT`` and ``GENESET``.
    ``GENESET`` is comma-separated; each entry is either a bare gene name
    (unweighted) or ``gene:weight`` (weighted).

    Parameters
    ----------
    gs_path
        Path to the ``.gs`` file.
    src_species, dst_species
        Optional species for homolog conversion; both ``None`` or both set.
    to_intersect
        Optional gene list to intersect every gene set with.

    Returns
    -------
    dict
        ``{trait: (gene_list, gene_weight_list)}``.
    """
    assert (src_species is None) == (dst_species is None), (
        "src_species and dst_species must be both None or not None"
    )
    if (src_species is not None and dst_species is not None) and (
        src_species != dst_species
    ):
        dict_map: Optional[Dict[str, str]] = load_homolog_mapping(
            src_species, dst_species
        )
    else:
        dict_map = None

    dict_gs: Dict[str, Tuple[List[str], List[float]]] = {}
    df_gs = pd.read_csv(gs_path, sep="\t")
    for _, (trait, gs) in df_gs.iterrows():
        gs_info = [g.split(":") for g in gs.split(",")]
        if np.all([len(g) == 1 for g in gs_info]):
            dict_weights = {g[0]: 1.0 for g in gs_info}
        elif np.all([len(g) == 2 for g in gs_info]):
            dict_weights = {g[0]: float(g[1]) for g in gs_info}
        else:
            raise ValueError(f"gene set {trait} contains genes with invalid format")

        if dict_map is not None:
            dict_weights = {
                dict_map[g]: w for g, w in dict_weights.items() if g in dict_map
            }
        if to_intersect is not None:
            inter = set(to_intersect)
            dict_weights = {g: w for g, w in dict_weights.items() if g in inter}

        gene_list = list(dict_weights.keys())
        dict_gs[trait] = (gene_list, [dict_weights[g] for g in gene_list])
    return dict_gs


def save_gs(gs_path: str, dict_gs: dict) -> None:
    """Write a ``.gs`` gene-set file.

    Parameters
    ----------
    gs_path
        Output path.
    dict_gs
        ``{trait: (gene_list, gene_weight_list)}`` or ``{trait: gene_list}``.
    """
    df_gs: Dict[str, list] = {"TRAIT": [], "GENESET": []}
    for trait in dict_gs:
        df_gs["TRAIT"].append(trait)
        if isinstance(dict_gs[trait], tuple):
            df_gs["GENESET"].append(
                ",".join(f"{g}:{w}" for g, w in zip(*dict_gs[trait]))
            )
        else:
            df_gs["GENESET"].append(",".join(dict_gs[trait]))
    pd.DataFrame(df_gs).to_csv(gs_path, sep="\t", index=False)


def munge_gs(
    out_path: str,
    df_zscore: Optional[pd.DataFrame] = None,
    df_pval: Optional[pd.DataFrame] = None,
    weight: str = "zscore",
    fdr: Optional[float] = None,
    fwer: Optional[float] = None,
    n_min: int = 100,
    n_max: int = 1000,
) -> Dict[str, Tuple[List[str], List[float]]]:
    """Build a ``.gs`` gene-set file from MAGMA z-score / p-value tables.

    Faithful to ``scdrs munge-gs``: for each trait (column) select the
    top genes (by smallest p-value), gate the size by ``fdr`` / ``fwer``
    or ``n_min`` / ``n_max``, and weight the selected genes.

    Parameters
    ----------
    out_path
        Output ``.gs`` path.
    df_zscore
        Gene-by-trait z-score table (index = gene names).
    df_pval
        Gene-by-trait p-value table.  Exactly one of ``df_zscore`` /
        ``df_pval`` should be supplied; the other is derived.
    weight
        ``zscore`` (z-score weights) or ``uniform`` (all-ones).
    fdr, fwer
        If set, select genes passing the BH-FDR / Bonferroni-FWER
        threshold (clipped to ``[n_min, n_max]``).
    n_min, n_max
        Min / max number of genes per gene set.

    Returns
    -------
    dict
        ``{trait: (gene_list, gene_weight_list)}`` (also written to disk).
    """
    from statsmodels.stats.multitest import multipletests

    assert weight in ("zscore", "uniform"), "weight must be 'zscore' or 'uniform'"
    assert (df_zscore is None) != (df_pval is None), (
        "Supply exactly one of df_zscore / df_pval"
    )
    if df_zscore is None:
        df_zscore = pd.DataFrame(
            pval2zsc(df_pval.values), index=df_pval.index, columns=df_pval.columns
        )
    if df_pval is None:
        df_pval = pd.DataFrame(
            zsc2pval(df_zscore.values), index=df_zscore.index, columns=df_zscore.columns
        )

    dict_gs: Dict[str, Tuple[List[str], List[float]]] = {}
    for trait in df_zscore.columns:
        v_z = df_zscore[trait].astype(float)
        v_p = df_pval[trait].astype(float)
        order = np.argsort(v_p.values)
        genes_sorted = v_z.index.values[order]

        if fdr is not None:
            n_sig = int((multipletests(v_p.values, method="fdr_bh")[1] < fdr).sum())
        elif fwer is not None:
            n_sig = int(
                (multipletests(v_p.values, method="bonferroni")[1] < fwer).sum()
            )
        else:
            n_sig = n_max
        n_sel = int(np.clip(n_sig, n_min, n_max))
        n_sel = min(n_sel, len(genes_sorted))

        sel_genes = list(genes_sorted[:n_sel])
        if weight == "zscore":
            sel_weights = [float(v_z.loc[g]) for g in sel_genes]
        else:
            sel_weights = [1.0] * len(sel_genes)
        dict_gs[str(trait)] = (sel_genes, sel_weights)

    save_gs(out_path, dict_gs)
    return dict_gs


# ----------------------------------------------------------------------
# AnnData I/O
# ----------------------------------------------------------------------
def load_h5ad(
    h5ad_file: str,
    flag_filter_data: bool = False,
    flag_raw_count: bool = True,
):
    """Load an ``.h5ad`` file, optionally filtering and normalizing.

    Parameters
    ----------
    h5ad_file
        Path to the ``.h5ad`` file.
    flag_filter_data
        If ``True``, drop low-quality cells / genes
        (``min_genes=250`` / ``min_cells=50``).
    flag_raw_count
        If ``True``, size-factor normalize (1e4) and log1p transform.

    Returns
    -------
    anndata.AnnData
        The loaded single-cell data.
    """
    import scanpy as sc
    from anndata import read_h5ad

    adata = read_h5ad(h5ad_file)
    if np.isnan(adata.X.sum()):
        raise ValueError(
            "h5ad expression matrix should not contain NaN. Impute beforehand."
        )
    if (adata.X < 0).sum() > 0:
        raise ValueError(
            "h5ad expression matrix should not contain negative values; "
            "scDRS models the gene-level log mean-variance relationship."
        )
    if flag_filter_data:
        sc.pp.filter_cells(adata, min_genes=250)
        sc.pp.filter_genes(adata, min_cells=50)
    if flag_raw_count:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
    return adata


# ----------------------------------------------------------------------
# z-score <-> p-value
# ----------------------------------------------------------------------
def zsc2pval(zsc):
    """One-sided p-value from a z-score (accurate to ~``zsc=36``)."""
    import scipy as sp

    return sp.stats.norm.cdf(-np.asarray(zsc, dtype=float))


def pval2zsc(pval):
    """Z-score from a one-sided p-value (accurate to ~``zsc=36``)."""
    import scipy as sp

    return -sp.stats.norm.ppf(np.asarray(pval, dtype=float))
