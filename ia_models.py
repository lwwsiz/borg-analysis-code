"""Pure-array IA/IgG context mathematics; no loader, CLI, or filesystem access.

The calling real-data gate owns source authorization, declared LVEF percent units,
frozen probe membership, paired visits, and the source-specific n=33 contract.
This module does not open data or enforce an arbitrary minimum sample size.
Array-valued intermediate helpers are for in-memory computation only. The public
model report contains aggregate statistics, never sample identifiers or vectors.

The sole primary model is LVEF6 ~ 1 + delta_f9_z + LVEF0 + F9_0.
Delta is always month 6 minus baseline. Sensitivities only add the two specified
PAN and/or inflammation covariates. They are descriptive, not separate primary
claims. All fitted models use HC3 and two-sided t(n-p) 95% intervals, no P values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


PRIMARY_COLUMNS = ("intercept", "delta_f9_z", "lvef0", "f9_0")
PAN_COLUMNS = ("pan0", "delta_pan")
INFLAMMATION_COLUMNS = ("inflammation0", "delta_inflammation")
ALLOWED_DESIGNS = (
    PRIMARY_COLUMNS,
    PRIMARY_COLUMNS + PAN_COLUMNS,
    PRIMARY_COLUMNS + INFLAMMATION_COLUMNS,
    PRIMARY_COLUMNS + PAN_COLUMNS + INFLAMMATION_COLUMNS,
)
LEVERAGE_GAP_TOLERANCE = 1e-10


class ModelBlocked(ValueError):
    """Sanitized input/numerical contract failure; never includes sample values."""


def _vector(values, *, allow_missing=True):
    try:
        out = np.array(values, dtype=float, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelBlocked("Expected a numeric one-dimensional array.") from exc
    if out.ndim != 1:
        raise ModelBlocked("Expected a numeric one-dimensional array.")
    if np.any(np.isinf(out)) or (not allow_missing and not np.all(np.isfinite(out))):
        raise ModelBlocked("Infinite or prohibited missing input; no data repair is allowed.")
    return out


@dataclass(frozen=True)
class AffineReference:
    """One immutable cohort mean and sample SD (ddof=1), with no row data."""

    n: int
    mean: float
    sample_sd: float

    def __post_init__(self):
        if (not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 2
                or not np.isfinite(self.mean) or not np.isfinite(self.sample_sd)
                or self.sample_sd <= 0):
            raise ModelBlocked("A finite positive sample SD and at least two reference values are required.")

    def apply(self, values):
        """Return an in-memory vector; preserve NaN, never refit the reference."""
        values = _vector(values)
        with np.errstate(over="ignore", invalid="ignore"):
            result = (values - self.mean) / self.sample_sd
        if np.any(np.isfinite(values) & ~np.isfinite(result)):
            raise ModelBlocked("Nonfinite affine result; no data repair is allowed.")
        return result

    def aggregate(self):
        return {"n": self.n, "mean": float(self.mean),
                "sample_sd": float(self.sample_sd), "sd_ddof": 1}


def freeze_reference(values):
    """Freeze all supplied finite reference values, without complete-case repair."""
    values = _vector(values, allow_missing=False)
    if len(values) < 2:
        raise ModelBlocked("A sample SD requires at least two reference values.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
    return AffineReference(len(values), mean, sd)


def freeze_baseline_program_reference(baseline_scores, dcm_baseline_mask):
    """Use exactly 40 DCM baselines, excluding the eight clinical controls.

    This is program-level affine rescaling of a log2 gene-equal mean. It is not
    per-gene z scoring, a new biological score definition, or cohort calibration.
    The caller supplies the 48-baseline inventory and an explicit boolean mask.
    Missingness in a selected DCM baseline blocks freezing; the denominator is
    never changed. Control values are never used in the reference statistics.
    """
    scores = _vector(baseline_scores)
    mask = np.asarray(dcm_baseline_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1 or mask.shape != scores.shape:
        raise ModelBlocked("An explicit aligned boolean DCM-baseline mask is required.")
    if len(scores) != 48 or int(np.sum(mask)) != 40:
        raise ModelBlocked("The frozen reference inventory requires 40 DCM baselines and eight excluded controls.")
    return freeze_reference(scores[mask])


def linear_probe_program_score(linear_probe_by_sample, gene_probe_indices: Mapping[str, Sequence[int]]):
    """One log2, all frozen probes averaged per gene, then genes equally weighted.

    Returns an in-memory sample vector. NaN in any selected probe makes its
    gene/sample and hence that program/sample NaN. No nanmean or denominator
    adaptation is permitted. A caller must verify positive-linear input provenance;
    numeric magnitudes cannot identify whether data were already logged.
    """
    try:
        values = np.array(linear_probe_by_sample, dtype=float, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelBlocked("Expected a numeric probe-by-sample matrix.") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ModelBlocked("Expected a nonempty probe-by-sample matrix.")
    if np.any(np.isinf(values)) or np.any(np.isfinite(values) & (values <= 0)):
        raise ModelBlocked("Linear probe input must be positive or explicitly missing.")
    if not isinstance(gene_probe_indices, Mapping) or not gene_probe_indices:
        raise ModelBlocked("A nonempty frozen gene-to-probe map is required.")
    selected = []
    for gene, indexes in gene_probe_indices.items():
        if not isinstance(gene, str) or not gene or gene.strip() != gene:
            raise ModelBlocked("Frozen gene names must be nonempty exact strings.")
        try:
            indexes = tuple(indexes)
        except TypeError as exc:
            raise ModelBlocked("Every gene requires a sequence of frozen probe indices.") from exc
        if (not indexes or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                               for i in indexes)
                or len(set(indexes)) != len(indexes)
                or any(i < 0 or i >= values.shape[0] for i in indexes)):
            raise ModelBlocked("Every gene requires distinct, in-range frozen probe indices.")
        selected.append(np.asarray(indexes, dtype=int))
    logged = np.log2(values)
    # Ordinary mean deliberately propagates any missing frozen probe/gene.
    gene_scores = np.vstack([np.mean(logged[indexes, :], axis=0) for indexes in selected])
    return np.mean(gene_scores, axis=0)


def _finite_scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _validate_design(x, y, names):
    names = tuple(names)
    if names not in ALLOWED_DESIGNS:
        raise ModelBlocked("Design differs from a frozen primary or descriptive sensitivity design.")
    try:
        x = np.array(x, dtype=float, copy=True)
        y = np.array(y, dtype=float, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelBlocked("Expected numeric design and outcome arrays.") from exc
    if x.ndim != 2 or y.ndim != 1 or x.shape != (len(y), len(names)):
        raise ModelBlocked("Design, outcome, and frozen column names are not aligned.")
    if len(y) and np.any(x[:, 0] != 1):
        raise ModelBlocked("The frozen first column must be an unchanged intercept.")
    return x, y, names


def _fit_numeric(x, y, names):
    """Internal matrices never leave the aggregate reporting boundary."""
    x, y, names = _validate_design(x, y, names)
    n, p = x.shape
    out = {"status": "UNIDENTIFIABLE", "n": n, "p": p, "df_residual": n - p,
           "columns": list(names), "rank": None, "rank_tolerance": None,
           "condition_number_raw": None, "condition_number_column_standardized": None,
           "max_leverage": None, "leverage_gap_tolerance": LEVERAGE_GAP_TOLERANCE,
           "finite_design_and_outcome": bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y))),
           "diagnostic_standardization_only": True, "covariates_dropped": 0,
           "caution_flags": {"standardized_condition_above_1000": None,
                             "max_leverage_above_0_5": None,
                             "descriptive_only_no_design_change": True}}
    if n == 0 or not out["finite_design_and_outcome"]:
        out["reason"] = "Empty or nonfinite frozen design/outcome."
        return out, None
    try:
        singular = np.linalg.svd(x, compute_uv=False)
        tolerance = max(x.shape) * np.finfo(float).eps * singular[0]
        out["rank"] = int(np.sum(singular > tolerance))
        out["rank_tolerance"] = float(tolerance)
        out["condition_number_raw"] = _finite_scalar(np.linalg.cond(x))
        scaled = x.copy()
        # For diagnostics only; the fitted design and its rank are not repaired.
        for j in range(1, p):
            scale = np.std(x[:, j], ddof=1) if n > 1 else 0.0
            if np.isfinite(scale) and scale > 0:
                scaled[:, j] = (x[:, j] - np.mean(x[:, j])) / scale
        standardized_condition = np.linalg.cond(scaled)
        out["condition_number_column_standardized"] = _finite_scalar(standardized_condition)
        out["caution_flags"]["standardized_condition_above_1000"] = bool(standardized_condition > 1000)
        if out["rank"] < p or n - p <= 0:
            out["reason"] = "Rank-deficient frozen design or nonpositive residual degrees of freedom."
            return out, None
        # Reduced QR with an ordinary inverse solve; no pseudoinverse, ridge,
        # column selection, rank correction, or outcome-informed adaptation.
        q, r = np.linalg.qr(x, mode="reduced")
        inverse = np.linalg.solve(r, q.T)
        beta = inverse @ y
        residual = y - x @ beta
        leverage = np.sum(q * q, axis=1)
        out["max_leverage"] = _finite_scalar(np.max(leverage))
        out["caution_flags"]["max_leverage_above_0_5"] = bool(np.max(leverage) > 0.5)
        if (not np.all(np.isfinite(leverage)) or np.any(leverage < -LEVERAGE_GAP_TOLERANCE)
                or np.any(1.0 - leverage <= LEVERAGE_GAP_TOLERANCE)):
            out["reason"] = "HC3 is undefined or numerically unstable because a leverage gap is nonpositive or too small."
            return out, None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weighted_inverse = inverse * (residual / (1.0 - leverage))[None, :]
            covariance = weighted_inverse @ weighted_inverse.T
        if not all(np.all(np.isfinite(v)) for v in (beta, residual, covariance)):
            out["reason"] = "Nonfinite coefficient, residual, or HC3 covariance; inference withheld."
            return out, None
        focal = names.index("delta_f9_z")
        variance = float(covariance[focal, focal])
        if variance < 0:
            out["reason"] = "Negative HC3 focal variance; no variance repair is allowed."
            return out, None
        estimate = float(beta[focal])
        se = float(np.sqrt(variance))
        critical = float(student_t.ppf(0.975, n - p))
        lower, upper = estimate - critical * se, estimate + critical * se
        if not all(np.isfinite(v) for v in (estimate, se, critical, lower, upper)):
            out["reason"] = "Nonfinite uncertainty calculation; inference withheld."
            return out, None
    except np.linalg.LinAlgError:
        out["reason"] = "Linear algebra failed on the unchanged frozen design; inference withheld."
        return out, None
    out.update(status="ESTIMATED", estimate=estimate, hc3_standard_error=se,
               confidence_interval_95_hc3_t=[float(lower), float(upper)], t_critical=critical,
               interval_reference="Two-sided Student t with df=n-p; HC3 is a small-sample approximation, not exact coverage.",
               units="LVEF percentage points per one primary-complete-cohort sample SD of month6-minus-baseline F9 change")
    internal = {"beta": beta, "residual": residual, "covariance": covariance,
                "projection": q @ q.T}
    return out, internal


def fit_hc3(x, y, names=PRIMARY_COLUMNS):
    """Return only a frozen model's focal result and aggregate diagnostics."""
    return _fit_numeric(x, y, names)[0]


def _empty_model_report(names):
    """Mark an unavailable required-field subset without invoking a fit."""
    p = len(names)
    return {"status": "UNIDENTIFIABLE", "n": 0, "p": p, "df_residual": -p,
            "columns": list(names), "rank": 0, "rank_tolerance": None,
            "condition_number_raw": None, "condition_number_column_standardized": None,
            "max_leverage": None, "leverage_gap_tolerance": LEVERAGE_GAP_TOLERANCE,
            "finite_design_and_outcome": True, "diagnostic_standardization_only": True,
            "covariates_dropped": 0, "fit_attempted": False,
            "reason": "No complete rows for the required fields; associated measurements are unavailable on a common subset.",
            "caution_flags": {"standardized_condition_above_1000": None,
                              "max_leverage_above_0_5": None,
                              "descriptive_only_no_design_change": True}}


def _parameterization_identity(x, y, lvef0, primary, primary_internal):
    if primary_internal is None:
        return {"status": "NOT_AVAILABLE", "reason": "Primary model is unidentifiable."}
    changed, internal = _fit_numeric(x, y - lvef0, PRIMARY_COLUMNS)
    if internal is None:
        return {"status": "NOT_AVAILABLE", "reason": "Numerical identity calculation was unidentifiable."}
    expected_beta = primary_internal["beta"].copy()
    expected_beta[PRIMARY_COLUMNS.index("lvef0")] -= 1.0
    verified = (np.allclose(internal["beta"], expected_beta, rtol=1e-9, atol=1e-10)
                and np.allclose(internal["residual"], primary_internal["residual"], rtol=1e-9, atol=1e-10)
                and np.allclose(internal["covariance"], primary_internal["covariance"], rtol=1e-9, atol=1e-10))
    return {"status": "VERIFIED" if verified else "NUMERICAL_IDENTITY_NOT_VERIFIED",
            "absolute_focal_estimate_difference": float(abs(primary["estimate"] - changed["estimate"])),
            "absolute_hc3_se_difference": float(abs(primary["hc3_standard_error"] - changed["hc3_standard_error"])),
            "max_absolute_coefficient_identity_difference": float(np.max(np.abs(internal["beta"] - expected_beta))),
            "max_absolute_residual_difference": float(np.max(np.abs(internal["residual"] - primary_internal["residual"]))),
            "max_absolute_covariance_difference": float(np.max(np.abs(internal["covariance"] - primary_internal["covariance"]))),
            "alternative_effect_reported": False,
            "meaning": "Replacing LVEF6 by LVEF6-LVEF0 with unchanged LVEF0 adjustment is only a numerical parameterization identity."}


def analyze_arrays(lvef0, lvef6, f9_0, f9_6, *, pan0=None, pan6=None,
                   inflammation0=None, inflammation6=None):
    """Aggregate-only frozen primary, paired-subset sensitivities, and LOO.

    Inputs are aligned in-memory vectors, without sample identifiers. LVEF must
    already be explicitly declared in percent; this function never detects or
    converts units. F9 and PAN must use their frozen program definitions and the
    same scale at both visits. Inflammation is the unchanged nonnegative combined
    CD3+CD68 histological count in cells/mm^2, with no log or other transformation.
    NaN denotes missingness. Infinite values
    are blocked, never silently dropped. Real-data authorization is NOT supplied
    by calling this function; it belongs to a separate caller-controlled gate.

    Primary completeness uses only the four required visit/outcome arrays. Each
    sensitivity uses its required-field subset of that primary cohort and refits
    the primary reference on exactly that subset. The focal delta mean and SD
    remain fixed from the whole primary-complete cohort in every fit and deletion.
    """
    arrays = {"lvef0": _vector(lvef0), "lvef6": _vector(lvef6),
              "f9_0": _vector(f9_0), "f9_6": _vector(f9_6)}
    input_n = len(arrays["lvef0"])
    if any(len(a) != input_n for a in arrays.values()):
        raise ModelBlocked("Primary input vectors are not aligned.")
    for name in ("lvef0", "lvef6"):
        a = arrays[name]
        if np.any(np.isfinite(a) & ((a < 0) | (a > 100))):
            raise ModelBlocked("LVEF is outside the explicitly declared percent range; no automatic unit conversion.")
    for name, values in (("pan0", pan0), ("pan6", pan6),
                         ("inflammation0", inflammation0), ("inflammation6", inflammation6)):
        arrays[name] = np.full(input_n, np.nan) if values is None else _vector(values)
        if len(arrays[name]) != input_n:
            raise ModelBlocked("Sensitivity input vectors are not aligned.")
        if name in ("inflammation0", "inflammation6") and np.any(arrays[name] < 0):
            raise ModelBlocked("Inflammation requires unchanged nonnegative histological counts.")
    primary_mask = np.logical_and.reduce([np.isfinite(arrays[k]) for k in ("lvef0", "lvef6", "f9_0", "f9_6")])
    data = {k: v[primary_mask] for k, v in arrays.items()}
    n = int(np.sum(primary_mask))
    with np.errstate(over="ignore", invalid="ignore"):
        delta = data["f9_6"] - data["f9_0"]
    reference = freeze_reference(delta)
    delta_z = reference.apply(delta)
    x = np.column_stack((np.ones(n), delta_z, data["lvef0"], data["f9_0"]))
    y = data["lvef6"]
    primary, internal = _fit_numeric(x, y, PRIMARY_COLUMNS)
    sensitivities = {}
    augmented_fits_attempted = 0
    # The same complete subset has one M0 result. With all fields complete this
    # reuses the sole primary fit, so there are exactly four substantive fits.
    # Byte keys are transient boolean masks, never sample IDs or report fields.
    primary_reference_cache = {np.ones(n, dtype=bool).tobytes(): primary}
    for label, variables in (("pan", ("pan0", "pan6")),
                             ("inflammation", ("inflammation0", "inflammation6")),
                             ("pan_and_inflammation", ("pan0", "pan6", "inflammation0", "inflammation6"))):
        mask = np.logical_and.reduce([np.isfinite(data[v]) for v in variables])
        additional, extra_names = [], ()
        with np.errstate(over="ignore", invalid="ignore"):
            if "pan0" in variables:
                additional.extend((data["pan0"][mask], (data["pan6"] - data["pan0"])[mask]))
                extra_names += PAN_COLUMNS
            if "inflammation0" in variables:
                additional.extend((data["inflammation0"][mask], (data["inflammation6"] - data["inflammation0"])[mask]))
                extra_names += INFLAMMATION_COLUMNS
        augmented = np.column_stack((x[mask], *additional))
        subset_n = int(np.sum(mask))
        if subset_n == 0:
            subset_primary = _empty_model_report(PRIMARY_COLUMNS)
            augmented_report = _empty_model_report(PRIMARY_COLUMNS + extra_names)
        else:
            subset_key = mask.tobytes()
            if subset_key not in primary_reference_cache:
                primary_reference_cache[subset_key] = fit_hc3(x[mask], y[mask], PRIMARY_COLUMNS)
            subset_primary = primary_reference_cache[subset_key]
            augmented_report = fit_hc3(augmented, y[mask], PRIMARY_COLUMNS + extra_names)
            augmented_fits_attempted += 1
        sensitivities[label] = {
            "interpretation": "DESCRIPTIVE_ONLY", "n_complete": subset_n,
            "excluded_from_primary_for_missing_required_fields": n - subset_n,
            "primary_reference_same_subset": subset_primary,
            "augmented": augmented_report,
            "primary_delta_reference_reused": True,
        }
    loo_estimates = []
    for deletion in range(n):
        retain = np.arange(n) != deletion
        deletion_result = fit_hc3(x[retain], y[retain], PRIMARY_COLUMNS)
        if deletion_result["status"] == "ESTIMATED":
            loo_estimates.append(deletion_result["estimate"])
    primary_sign = np.sign(primary["estimate"]) if primary["status"] == "ESTIMATED" else None
    loo = {"attempted": n, "estimated": len(loo_estimates),
           "unidentifiable": n - len(loo_estimates),
           "coefficient_min": min(loo_estimates) if loo_estimates else None,
           "coefficient_max": max(loo_estimates) if loo_estimates else None,
           "positive_count": int(sum(v > 0 for v in loo_estimates)),
           "negative_count": int(sum(v < 0 for v in loo_estimates)),
           "zero_count": int(sum(v == 0 for v in loo_estimates)),
           "same_sign_as_primary_count": int(sum(np.sign(v) == primary_sign for v in loo_estimates)) if primary_sign is not None else None,
           "primary_delta_reference_reused": True, "individual_deletion_results_exported": False}
    return {"status": "AGGREGATE_MATHEMATICS_COMPLETE", "input_n": input_n,
            "primary_complete_n": n, "primary_missing_excluded_n": input_n - n,
            "focal_delta_reference": reference.aggregate(), "primary": primary,
            "sensitivities": sensitivities, "leave_one_out": loo,
            "lvef_change_parameterization_identity": _parameterization_identity(x, y, data["lvef0"], primary, internal),
            "primary_model_count": 1, "descriptive_sensitivity_count": 3,
            "actual_model_evaluations": (1 + augmented_fits_attempted
                                         + len(primary_reference_cache) - 1
                                         + n + int(internal is not None)),
            "execution_inventory": {"full_primary_fits": 1, "descriptive_augmented_fits": augmented_fits_attempted,
                                    "distinct_primary_subset_refits": len(primary_reference_cache) - 1,
                                    "loo_attempted": n,
                                    "parameterization_identity_attempted": int(internal is not None)},
            "sample_vectors_exported": False, "sample_identifiers_accepted": False,
            "causal_or_clinical_utility_claim": False}
