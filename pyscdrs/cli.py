"""Thin command-line interface for pyscdrs.

Mirrors the most-used scDRS CLI subcommands:

* ``pyscdrs compute-score`` — preprocess + score cells for every trait in
  a ``.gs`` file, writing ``<trait>.score.gz`` (and ``.full_score.gz``).
* ``pyscdrs perform-downstream`` — group / correlation / gene analyses on
  an existing ``.full_score.gz`` file.

This is intentionally minimal; the importable API
(:func:`pyscdrs.score_cell`, :func:`pyscdrs.preprocess`, ...) is the main
interface.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


def _cmd_compute_score(args: argparse.Namespace) -> None:
    from anndata import read_h5ad

    from .io import load_gs
    from .preprocess import preprocess
    from .score import score_cell

    adata = read_h5ad(args.h5ad_file)
    cov = None
    if args.cov_file is not None:
        cov = pd.read_csv(args.cov_file, sep="\t", index_col=0)
    preprocess(adata, cov=cov, n_mean_bin=args.n_mean_bin, n_var_bin=args.n_var_bin)

    dict_gs = load_gs(args.gs_file)
    os.makedirs(args.out_folder, exist_ok=True)
    for trait, (gene_list, gene_weight) in dict_gs.items():
        df_res = score_cell(
            adata,
            gene_list,
            gene_weight=gene_weight,
            n_ctrl=args.n_ctrl,
            weight_opt=args.weight_opt,
            return_ctrl_norm_score=args.flag_full_score,
            random_seed=args.random_seed,
            verbose=True,
        )
        cols = ["raw_score", "norm_score", "mc_pval", "pval", "nlog10_pval", "zscore"]
        df_res[cols].to_csv(
            os.path.join(args.out_folder, f"{trait}.score.gz"), sep="\t",
            compression="gzip",
        )
        if args.flag_full_score:
            df_res.to_csv(
                os.path.join(args.out_folder, f"{trait}.full_score.gz"), sep="\t",
                compression="gzip",
            )
        print(f"# pyscdrs compute-score: wrote results for trait '{trait}'")


def _cmd_perform_downstream(args: argparse.Namespace) -> None:
    from anndata import read_h5ad

    from .downstream import (
        downstream_corr_analysis,
        downstream_gene_analysis,
        downstream_group_analysis,
    )

    adata = read_h5ad(args.h5ad_file)
    df_full_score = pd.read_csv(args.full_score_file, sep="\t", index_col=0)
    os.makedirs(args.out_folder, exist_ok=True)
    stem = os.path.basename(args.full_score_file).replace(".full_score.gz", "")

    if args.group_analysis is not None:
        import scanpy as sc

        if "connectivities" not in adata.obsp:
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
        dict_res = downstream_group_analysis(
            adata, df_full_score, group_cols=args.group_analysis
        )
        for col, df_res in dict_res.items():
            df_res.to_csv(
                os.path.join(args.out_folder, f"{stem}.scdrs_group.{col}"), sep="\t"
            )
            print(f"# pyscdrs perform-downstream: wrote group analysis for '{col}'")
    if args.corr_analysis is not None:
        df_res = downstream_corr_analysis(
            adata, df_full_score, var_cols=args.corr_analysis
        )
        df_res.to_csv(
            os.path.join(args.out_folder, f"{stem}.scdrs_cell_corr"), sep="\t"
        )
        print("# pyscdrs perform-downstream: wrote correlation analysis")
    if args.gene_analysis:
        df_res = downstream_gene_analysis(adata, df_full_score)
        df_res.to_csv(os.path.join(args.out_folder, f"{stem}.scdrs_gene"), sep="\t")
        print("# pyscdrs perform-downstream: wrote gene analysis")


def main(argv: Optional[List[str]] = None) -> None:
    """pyscdrs command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="pyscdrs", description="pyscdrs: single-cell disease-relevance scoring"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("compute-score", help="preprocess and score cells")
    p_score.add_argument("--h5ad-file", required=True)
    p_score.add_argument("--gs-file", required=True)
    p_score.add_argument("--out-folder", required=True)
    p_score.add_argument("--cov-file", default=None)
    p_score.add_argument("--n-ctrl", type=int, default=1000)
    p_score.add_argument("--n-mean-bin", type=int, default=20)
    p_score.add_argument("--n-var-bin", type=int, default=20)
    p_score.add_argument("--weight-opt", default="vs")
    p_score.add_argument("--random-seed", type=int, default=0)
    p_score.add_argument("--flag-full-score", action="store_true")
    p_score.set_defaults(func=_cmd_compute_score)

    p_down = sub.add_parser("perform-downstream", help="downstream analyses")
    p_down.add_argument("--h5ad-file", required=True)
    p_down.add_argument("--full-score-file", required=True)
    p_down.add_argument("--out-folder", required=True)
    p_down.add_argument("--group-analysis", nargs="+", default=None)
    p_down.add_argument("--corr-analysis", nargs="+", default=None)
    p_down.add_argument("--gene-analysis", action="store_true")
    p_down.set_defaults(func=_cmd_perform_downstream)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
