"""Selected BORG scoring and clinical model functions.

Function bodies are copied unchanged from the analysis source. File acquisition,
internal manifests and execution orchestration are omitted. Caller-supplied data
must have the source-defined scale, membership, visit labels and units.
The name lvef12 denotes the original month-12-labelled LVEF field; it does not
independently establish the time of the LVEF measurement.
"""
import collections
import math
import numpy as np
from scipy.stats import t as student_t

PRIMARY_FIELDS = ('lvef0', 'lvef3', 'lvef12', 'treatment', 'f9_0', 'delta_f9')
BASE_COVARIATES = ('delta_f9_z', 'lvef3', 'lvef0', 'f9_0')
EXPECTED_COUNTS = {'clinical_people': 47, 'baseline_month3_pairs': 43,
                   'landmark_named_lvef3_lvef12': 36, 'main_complete': 36,
                   'hr_complete': 35, 'main_treatment_levels': 3,
                   'named_lvef12_available_all_clinical': 40,
                   'pairs_missing_named_lvef12': 7, 'pairs_missing_lvef3': 0}

class AnalysisBlocked(RuntimeError):
    """Message must describe a reason without IDs or patient values."""

def finite(v):
    try:
        return v is not None and str(v).strip() != '' and math.isfinite(float(v))
    except (ValueError, TypeError):
        return False

def fixed_score(expression, genes, key):
    if not genes:
        return None
    values = [expression.get(g, {}).get(key) for g in genes]
    if not all(finite(v) for v in values):
        return None
    # Genes are sorted by the frozen metadata, keeping floating-point accumulation deterministic.
    return math.fsum(float(v) for v in values) / len(values)

def make_cohort(clinical, sample_keys, expression, coverage):
    """IDs remain internal. No LOCF field is accepted by this function."""
    keys = set(sample_keys)
    pairs = {k for k in clinical if (k, 'A') in keys and (k, 'B') in keys}
    preliminary = {k for k in pairs if finite(clinical[k].get('LVEF_3mo')) and finite(clinical[k].get('LVEF_12mo'))}
    people = []; missing = collections.Counter()
    aliases = {'F9': 'f9', 'PAN416': 'pan'}
    for key in sorted(preliminary):
        source = clinical[key]
        d = {'lvef0': source.get('LVEF_bl'), 'lvef3': source.get('LVEF_3mo'),
             'lvef12': source.get('LVEF_12mo'), 'treatment': source.get('txgroup'),
             'hr0': source.get('HR_bl')}
        d['delta_hr'] = source.get('HR_3mo') - source.get('HR_bl') if finite(source.get('HR_3mo')) and finite(source.get('HR_bl')) else None
        for name, short in aliases.items():
            genes = coverage[name]['available_genes']
            zero = fixed_score(expression, genes, (key, 'A'))
            three = fixed_score(expression, genes, (key, 'B'))
            d[short + '_0'] = zero
            d['delta_' + short] = three - zero if finite(zero) and finite(three) else None
        absent = [f for f in PRIMARY_FIELDS if not finite(d.get(f))]
        if absent:
            missing.update(absent)
        else:
            people.append(d)
    hr = [d for d in people if finite(d.get('hr0')) and finite(d.get('delta_hr'))]
    flow = {'clinical_people': len(clinical), 'baseline_month3_pairs': len(pairs),
            'named_lvef12_available_all_clinical': sum(finite(d.get('LVEF_12mo')) for d in clinical.values()),
            'pairs_missing_named_lvef12': sum(not finite(clinical[k].get('LVEF_12mo')) for k in pairs),
            'pairs_missing_lvef3': sum(not finite(clinical[k].get('LVEF_3mo')) for k in pairs),
            'landmark_named_lvef3_lvef12': len(preliminary), 'main_complete': len(people),
            'primary_missing_fields_among_landmark': dict(missing),
            'hr_complete': len(hr), 'main_treatment_levels': len({d['treatment'] for d in people}),
            'locf_substitution_used': False, 'month12_expression_required': False}
    return people, flow

def freeze_scaling(people):
    scalings = {}
    for f in ('delta_f9',):
        values = [d[f] for d in people if finite(d.get(f))]
        if len(values) < 2:
            scalings[f] = {'status': 'UNIDENTIFIABLE', 'reason': 'Fewer than two complete score changes.'}
            continue
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        if not math.isfinite(sd) or sd <= np.finfo(float).eps * max(1., abs(mean)):
            scalings[f] = {'status': 'UNIDENTIFIABLE', 'reason': 'Zero or numerically vanishing score-change variation.'}
            continue
        scalings[f] = {'status': 'FIXED', 'n': len(values), 'mean': mean, 'sample_sd_ddof1': sd}
        for d in people:
            if finite(d.get(f)):
                d[f + '_z'] = (d[f] - mean) / sd
    return scalings

def design(people, covariates, levels):
    if len(levels) < 2:
        raise AnalysisBlocked('Fewer than two treatment categories; frozen adjustment is not identifiable.')
    names = ['intercept'] + list(covariates) + ['treatment_contrast_' + str(i) for i in range(1, len(levels))]
    if any(not all(finite(d.get(f)) for f in covariates + ('lvef12', 'treatment')) for d in people):
        raise AnalysisBlocked('Model receives incomplete required fields; no automatic missing-data repair.')
    if any(d['treatment'] not in levels for d in people):
        raise AnalysisBlocked('An unfrozen treatment category appeared.')
    x = np.asarray([[1.] + [float(d[f]) for f in covariates] + [float(d['treatment'] == level) for level in levels[1:]] for d in people], dtype=float)
    y = np.asarray([d['lvef12'] for d in people], dtype=float)
    return x, y, names

def _safe_float(value):
    return float(value) if math.isfinite(float(value)) else None

def fit_hc3(x, y, names, focal='delta_f9_z'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, p = x.shape
    out = {'n': int(n), 'parameters_including_intercept': int(p), 'residual_df': int(n - p),
           'covariates': list(names), 'focal_parameter': focal, 'status': 'UNIDENTIFIABLE'}
    if n == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        out['reason'] = 'Empty or nonfinite design.'; return out
    singular = np.linalg.svd(x, compute_uv=False)
    tol = max(x.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tol))
    out['rank'] = rank; out['rank_tolerance'] = float(tol)
    out['condition_number_raw'] = _safe_float(np.linalg.cond(x))
    scaled = x.copy()
    for j in range(1, p):
        std = np.std(x[:, j], ddof=1) if n > 1 else 0.
        if std > 0:
            scaled[:, j] = (x[:, j] - np.mean(x[:, j])) / std
    out['condition_number_column_standardized'] = _safe_float(np.linalg.cond(scaled))
    if n <= p or rank < p:
        out['reason'] = 'Insufficient residual degrees of freedom or rank-deficient frozen design.'; return out
    inverse = np.linalg.pinv(x)
    beta = inverse @ y
    residual = y - x @ beta
    hat = np.sum(x * inverse.T, axis=1)
    out['leverage_max'] = float(np.max(hat))
    if np.any(1 - hat <= 1e-10):
        out['reason'] = 'HC3 undefined or numerically unstable because leverage approaches one.'; return out
    adjusted = residual / (1 - hat)
    with np.errstate(over='ignore', invalid='ignore'):
        covariance = (inverse * adjusted) @ (inverse * adjusted).T
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance)):
        out['reason'] = 'Nonfinite coefficient or covariance after numerical calculation; inference withheld.'; return out
    index = names.index(focal)
    se = float(np.sqrt(max(0., covariance[index, index])))
    estimate = float(beta[index]); critical = float(student_t.ppf(.975, n - p))
    if not all(math.isfinite(v) for v in [estimate, se, critical, estimate-critical*se, estimate+critical*se]):
        out['reason'] = 'Nonfinite uncertainty calculation; inference withheld.'; return out
    out.update(status='ESTIMATED', estimate=estimate, hc3_standard_error=se,
               confidence_interval_95_hc3_t=[estimate - critical * se, estimate + critical * se],
               interval_reference='Student t, residual df n-p; HC3 small-sample approximation, not exact coverage',
               t_critical=critical,
               units='LVEF percentage points per 1 primary-cohort sample SD of early program change')
    out['warnings'] = []
    cond = out['condition_number_column_standardized']
    if cond is None or cond > 1000:
        out['warnings'].append('Severe standardized design conditioning; no covariate has been dropped.')
    if out['leverage_max'] > .5:
        out['warnings'].append('High leverage; HC3 and leave-one-out stability require cautious interpretation.')
    return out

def fit_people(people, covariates, levels, focal='delta_f9_z'):
    required = covariates + ('lvef12', 'treatment')
    complete = [d for d in people if all(finite(d.get(f)) for f in required)]
    if not complete:
        return {'status': 'UNIDENTIFIABLE', 'n': 0, 'reason': 'No field-complete people for the frozen model.'}
    try:
        x, y, names = design(complete, covariates, levels)
    except AnalysisBlocked as e:
        return {'status': 'UNIDENTIFIABLE', 'n': len(complete), 'reason': str(e)}
    result = fit_hc3(x, y, names, focal)
    result['excluded_by_required_field_missingness'] = len(people) - len(complete)
    return result

def correlation(people, a, b):
    complete = [(d[a], d[b]) for d in people if finite(d.get(a)) and finite(d.get(b))]
    if len(complete) < 3:
        return {'n': len(complete), 'pearson_r': None, 'status': 'UNIDENTIFIABLE'}
    values = np.asarray(complete, float)
    if np.any(np.std(values, axis=0) == 0):
        return {'n': len(complete), 'pearson_r': None, 'status': 'UNIDENTIFIABLE'}
    return {'n': len(complete), 'pearson_r': float(np.corrcoef(values.T)[0, 1]), 'status': 'ESTIMATED'}

def analyze(people, flow, enforce_counts=False):
    # Work on copies so repeated runs are identical and no caller state is changed.
    people = [dict(d) for d in people]
    if enforce_counts and any(flow.get(k) != v for k, v in EXPECTED_COUNTS.items()):
        return {'status': 'BLOCKED_SOURCE_CONTRACT', 'flow': flow, 'reason': 'At least one frozen aggregate cohort count changed; no effect model was fitted.'}
    scaling = freeze_scaling(people)
    levels = sorted({d['treatment'] for d in people})
    primary = fit_people(people, BASE_COVARIATES, levels)
    hr_people = [d for d in people if finite(d.get('hr0')) and finite(d.get('delta_hr'))]
    models = {'primary': primary,
              'hr_subset_base': fit_people(hr_people, BASE_COVARIATES, levels),
              'hr_augmented': fit_people(hr_people, BASE_COVARIATES + ('hr0', 'delta_hr'), levels),
              'pan_rna_content_augmented': fit_people(people, BASE_COVARIATES + ('pan_0', 'delta_pan'), levels)}
    loo_results = [fit_people(people[:i] + people[i+1:], BASE_COVARIATES, levels) for i in range(len(people))]
    coefficients = [r['estimate'] for r in loo_results if r['status'] == 'ESTIMATED']
    primary_sign = np.sign(primary['estimate']) if primary['status'] == 'ESTIMATED' else None
    loo = {'attempted': len(loo_results), 'estimated': len(coefficients), 'unidentifiable': len(loo_results) - len(coefficients),
           'coefficient_min': min(coefficients) if coefficients else None,
           'coefficient_max': max(coefficients) if coefficients else None,
           'positive_count': sum(v > 0 for v in coefficients), 'negative_count': sum(v < 0 for v in coefficients),
           'zero_count': sum(v == 0 for v in coefficients),
           'same_sign_as_primary_count': int(sum(np.sign(v) == primary_sign for v in coefficients)) if primary_sign is not None else None,
           'scaling': 'Primary-cohort mean and SD held fixed; treatment levels and covariates held fixed.',
           'individual_deletion_results_exported': False}
    equivalence = {'status': 'NOT_AVAILABLE', 'reason': 'Primary model was not identifiable.'}
    if primary['status'] == 'ESTIMATED':
        x, y, names = design(people, BASE_COVARIATES, levels)
        changed = fit_hc3(x, y - np.asarray([d['lvef3'] for d in people]), names)
        if changed['status'] == 'ESTIMATED':
            equivalence = {'status': 'CHECKED', 'absolute_focal_estimate_difference': abs(primary['estimate'] - changed['estimate']),
                           'absolute_hc3_se_difference': abs(primary['hc3_standard_error'] - changed['hc3_standard_error']),
                           'explanation': 'With the same LVEF3 adjustment, LVEF12 and LVEF12-LVEF3 yield the same focal coefficient and residuals.'}
    return {'status': 'ANALYSIS_COMPLETE_AGGREGATE_ONLY', 'flow': flow, 'scaling': scaling, 'models': models,
            'leave_one_out': loo, 'landmark_parameterization_equivalence': equivalence,
            'aggregate_correlations': {a + '__' + b: correlation(people, a, b) for a, b in [
                ('f9_0', 'delta_f9'), ('pan_0', 'delta_pan'), ('delta_f9', 'delta_pan'), ('f9_0', 'pan_0')]},
            'inference_inventory': {'primary_hypotheses': 1, 'association_model_fits': 4,
                                   'targeted_alternative_association_models': 2, 'matched_hr_reference_refits': 1,
                                   'leave_one_out_diagnostics': len(people), 'parameterization_identity_checks': 1,
                                   'p_values_reported': 0, 'sensitivity_intervals_are_independent_confirmatory_tests': False},
            'causal_or_clinical_utility_claim': False, 'individual_data_exported': False}
