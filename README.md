# BORG: selected analytical code

Selected code accompanying the BORG myocardial transcriptional-program manuscript
by LIN Wei-wei, LIU Yin and HE Yuan.

## Contents

- `clinical_models.py`: BORG gene-equal scoring, clinical cohort assembly, fixed
  change-score scaling, HC3 regression with residual-df t intervals, heart-rate
  and PAN adjustment, and patient-deletion diagnostics.
- `ia_models.py`: IA/IgG probe-to-gene scoring and clinical regression functions.
- `rna_atac_models.py`: donor-level RNA-ATAC HC3 regression and BH adjustment.
- `programs.json`: the BORG-matched gene lists: F9 (230), PAN (367), ECM (307).
  These are the BORG lists, not all dataset-specific or common RNA-ATAC projections.
- `requirements.txt`: the two external Python libraries used by these modules.

The selected function bodies retain the existing analytical implementations.
The clinical and RNA-ATAC modules omit source-data loaders, local paths and
execution orchestration. The IA/IgG module is the original array-based module.

## Use and scope

These are callable Python modules, not a complete end-to-end reproduction package.
For example, `clinical_models.fit_hc3(X, y, names)` fits a prepared design matrix;
`rna_atac_models.fit(x, y, covs)` fits a prepared donor-level comparison.
Input construction, units, selection criteria and dataset-specific projections
must follow the article and its Supplementary Methods.

The BORG outcome variable named `lvef12` refers to the original LVEF field labelled
month 12. Its name does not independently verify the measurement time.
The BORG `make_cohort` function retains the internal PAN key `PAN416`; its
`available_genes` must contain the 367 BORG-matched members, not all 416 source
members. Original treatment-category codes are not translated into drug names.
For RNA-ATAC analyses, the primary covariate is ischemic etiology; the depth
sensitivity additionally uses log10 mean RNA counts and log10 mean ATAC counts.
Apply `bh` only to the comparison families specified in the manuscript.

Original source datasets are obtained from the sources cited in the manuscript.
BORG supporting data: https://doi.org/10.1172/jci.insight.169720.
No patient-level records, expression matrices, cell-level objects, generated
results, figures, manuscript files or internal working records are included.
The repository does not include every supplementary-analysis implementation,
input-preparation pipeline, bootstrap procedure or reference-set generation step.
It does not claim fresh-environment or complete numerical reproduction.

Correspondence: HE Yuan, HeYuan@gdmu.edu.cn.
