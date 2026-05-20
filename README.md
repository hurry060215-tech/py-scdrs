# py-scdrs

`pyscdrs` is a clean, pure-Python reimplementation of
[**scDRS**](https://github.com/martinjzhang/scDRS) (single-cell
disease-relevance score) for the [omicverse](https://github.com/Starlitnightly/omicverse)
project.

scDRS (Zhang\*, Hou\*, et al. & Price, *Nature Genetics* 2022,
54:1572-1580) scores individual cells in scRNA-seq data for their
relevance to a disease or complex trait, using a polygenic gene set
derived from GWAS summary statistics (typically via MAGMA). The score is
a covariate-corrected, technical-variance-weighted average of disease-gene
expression, calibrated against Monte-Carlo control gene sets matched on
the gene-level mean-variance relationship.

This package is a faithful rewrite — with the same `random_seed` it
reproduces the original `scdrs` package's control gene sets, scores and
p-values **bit-for-bit** on the bundled toy data.

## Install

```bash
pip install pyscdrs
```

Pure Python — depends only on `numpy`, `scipy`, `pandas`, `anndata`,
`scanpy`, `scikit-misc`, `statsmodels` and `tqdm`. No R / rpy2.

## Quick start

```python
import anndata, pandas as pd
import pyscdrs as scdrs

# Load size-factor-normalized, log1p-transformed single-cell data
adata  = anndata.read_h5ad("toydata_mouse.h5ad")
df_cov = pd.read_csv("toydata_mouse.cov", sep="\t", index_col=0)

# 1. Preprocess: covariate correction + gene/cell stats + mean-var bins
scdrs.preprocess(adata, cov=df_cov)

# 2. Load a MAGMA-style .gs gene set and score cells
dict_gs = scdrs.load_gs("toydata_mouse.gs")
genes, weights = dict_gs["toydata_gs_mouse"]
df_res = scdrs.score_cell(
    adata, genes, gene_weight=weights, n_ctrl=1000, random_seed=0,
    return_ctrl_norm_score=True,
)
print(df_res[["raw_score", "norm_score", "pval", "zscore"]].head())

# 3. Downstream: cell-group association + heterogeneity
import scanpy as sc
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
dict_group = scdrs.downstream_group_analysis(
    adata, df_res, group_cols=["cell_type"]
)
```

## Public API

| Stage          | Function |
|----------------|----------|
| Preprocessing  | `preprocess`, `compute_stats`, `reg_out`, `category2dummy` |
| Scoring        | `score_cell` |
| Downstream     | `downstream_group_analysis`, `downstream_corr_analysis`, `downstream_gene_analysis`, `test_gearysc`, `gearys_c` |
| Gene-set / I/O | `load_gs`, `save_gs`, `munge_gs`, `load_h5ad`, `load_homolog_mapping`, `convert_species_name`, `zsc2pval`, `pval2zsc` |

## `score_cell` options

* `weight_opt` — raw-score weighting: `uniform`, `vs`
  (1/sqrt technical variance, default), `inv_std` (1/std), `od`
  (overdispersion score).
* `ctrl_match_key` — gene statistic for matching control genes
  (default `mean_var`).
* `n_ctrl` — number of Monte-Carlo control gene sets (default 1000).
* `random_seed` — governs the control gene sets; the same seed gives the
  same results as the original `scdrs`.

The output is a per-cell `DataFrame` with `raw_score`, `norm_score`,
`mc_pval` (per-cell Monte-Carlo p-value), `pval` (pooled empirical
p-value), `nlog10_pval` and `zscore`.

## Command line

```bash
pyscdrs compute-score --h5ad-file data.h5ad --gs-file trait.gs \
    --cov-file cov.tsv --out-folder out/ --n-ctrl 1000 --flag-full-score

pyscdrs perform-downstream --h5ad-file data.h5ad \
    --full-score-file out/trait.full_score.gz --out-folder out/ \
    --group-analysis cell_type --gene-analysis
```

## Parity with the original scDRS

`tests/test_parity.py` runs both `pyscdrs` and the upstream `scdrs`
package on scDRS's own bundled toy data with identical `random_seed` and
asserts agreement of `preprocess` (gene stats and mean-variance bins),
`score_cell` (`raw_score`, `norm_score`, `pval`, `mc_pval`, `zscore`) and
the downstream group / gene / correlation statistics. On the toy data the
agreement is bit-exact.

## License

MIT, same as the original scDRS. See `LICENSE`.
