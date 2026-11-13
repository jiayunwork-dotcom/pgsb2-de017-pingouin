# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import studentized_range, t

from .config import _no_rounding
from .effsize import compute_effsize
from .multicomp import _multicomp_triu, multicomp
from .utils import (
    _check_alternative,
    _check_dataframe,
    _flatten_list,
    _format_pairwise_matrix,
    _postprocess_dataframe,
)

__all__ = [
    "pairwise_ttests",
    "pairwise_tests",
    "ptests",
    "pairwise_tukey",
    "pairwise_gameshowell",
    "pairwise_corr",
]


def _pairwise_test(x, y, paired, parametric, alternative, correction, effsize, return_desc):
    """Compare two samples and return the output row of :py:func:`pairwise_tests` as a dict.

    Uses :py:func:`pingouin.ttest` if ``parametric``, otherwise :py:func:`pingouin.wilcoxon` or
    :py:func:`pingouin.mwu` for paired and unpaired samples, respectively.
    """
    from .nonparametric import mwu, wilcoxon
    from .parametric import ttest

    row = {"alternative": alternative}
    if return_desc:
        row.update(
            mean_A=np.nanmean(x),
            std_A=np.nanstd(x, ddof=1),
            mean_B=np.nanmean(y),
            std_B=np.nanstd(y, ddof=1),
        )
    if parametric:
        stat_name = "T"
        res = ttest(x, y, paired=paired, alternative=alternative, correction=correction)
        row["dof"] = float(res.at["T_test", "dof"])
        if alternative == "two-sided":
            row["BF10"] = res.at["T_test", "BF10"]
    elif paired:
        stat_name = "W_val"
        res = wilcoxon(x, y, alternative=alternative)
    else:
        stat_name = "U_val"
        res = mwu(x, y, alternative=alternative)
    row[stat_name] = res[stat_name].iat[0]
    row["p_unc"] = res["p_val"].iat[0]
    row[effsize] = compute_effsize(x=x, y=y, eftype=effsize, paired=paired)
    return row


def _padjust(stats, padjust, alpha=0.05):
    """Add the ``p_corr`` and ``p_adjust`` columns to a family of tests (in place) and return it.

    Nothing is added if ``padjust`` is None or ``'none'``, or if there is a single test.
    """
    if padjust is not None and padjust.lower() != "none" and stats.shape[0] > 1:
        stats["p_corr"] = multicomp(stats["p_unc"].to_numpy(), alpha=alpha, method=padjust)[1]
        stats["p_adjust"] = padjust
    return stats


def _group_arrays(data, by, dv):
    """Values of ``dv`` for each level of ``by``, as a dict of float arrays sorted by level.

    Levels are sorted with groupby and not np.unique, because the latter does not respect the
    custom sorting of a Categorical column. See https://github.com/raphaelvallat/pingouin/issues/111
    """
    grp = data.groupby(by, observed=True, sort=True)[dv]
    return {k: v.to_numpy(dtype=np.float64) for k, v in grp}


def _t_pvalue(tval, dof, alternative):
    """P-value of a T statistic with ``dof`` degrees of freedom, for the given tail."""
    if alternative == "greater":
        return t.sf(tval, dof)
    elif alternative == "less":
        return t.cdf(tval, dof)
    return 2 * t.sf(np.abs(tval), dof)


def _pooled_row(diff, se, dof, alternative, paired, x, y, effsize, desc=None):
    """Build one output row of a pooled pairwise test from an estimated contrast and its SE.

    The T statistic is the estimated difference divided by the pooled standard error, and the
    p-value uses the (possibly non-integer) pooled degrees of freedom. The effect size is still
    computed pairwise from the raw observations, exactly as in :py:func:`pairwise_tests`.
    ``desc`` optionally provides the (mean_A, std_A, mean_B, std_B) descriptive statistics.
    """
    tval = diff / se
    row = {
        "alternative": alternative,
        "T": tval,
        "dof": float(dof),
        "se": se,
        "p_unc": _t_pvalue(tval, dof, alternative),
        effsize: compute_effsize(x=x, y=y, eftype=effsize, paired=paired),
    }
    if desc is not None:
        row.update(zip(("mean_A", "std_A", "mean_B", "std_B"), desc))
    return row


def _cell_stats(data, dv, factors):
    """Per-cell mean, sample size and variance of a factorial between-subject design."""
    grp = data.groupby(factors, observed=True, sort=True)[dv]
    return grp.mean(), grp.count(), grp.var(numeric_only=True)


def _pooled_simple_between(data, dv, factor, effsize, alternative, return_desc):
    """Pooled comparisons for a single between-subject factor.

    The error term is the Within mean square of the one-way ANOVA
    (:py:func:`pingouin.anova`), with ``N - k`` degrees of freedom, and the standard errors are
    the Tukey-Kramer standard errors, so the T-values match :py:func:`pingouin.pairwise_tukey`.
    """
    grp = data.groupby(factor, observed=True, sort=True)[dv]
    means, n, gvars = grp.mean(numeric_only=True), grp.count(), grp.var(numeric_only=True)
    dof = n.sum() - len(n)
    mse = np.nansum((n - 1) * gvars) / dof
    groups = _group_arrays(data, factor, dv)
    rows = []
    for a, b in combinations(means.index, 2):
        se = np.sqrt(mse * (1 / n[a] + 1 / n[b]))
        desc = (
            (means[a], np.nanstd(groups[a], ddof=1), means[b], np.nanstd(groups[b], ddof=1))
            if return_desc
            else None
        )
        row = {"Contrast": factor, "A": a, "B": b, "Paired": False}
        row.update(
            _pooled_row(
                means[a] - means[b], se, dof, alternative, False, groups[a], groups[b],
                effsize, desc,
            )
        )
        rows.append(row)
    return rows


def _pooled_multiple_between(data, dv, factors, effsize, alternative, return_desc):
    """Pooled comparisons for two between-subject factors.

    The error term is the residual mean square of the full factorial model (with the
    interaction), whose degrees of freedom are the total sample size minus the number of cells.
    Main-effect contrasts compare estimated marginal means, i.e. unweighted averages of the cell
    means across the levels of the other factor. Interaction contrasts compare cell means within
    each level of the first factor, and use the same standard errors (and therefore T-values) as
    :py:func:`pingouin.pairwise_tukey`.
    """
    f1, f2 = factors
    data = data[[dv, f1, f2]].dropna()
    n_cells = data[f1].nunique() * data[f2].nunique()
    if data.groupby([f1, f2], observed=True).ngroups < n_cells:
        raise ValueError(
            "Each combination of the between-subject factors must have at least one "
            "observation. Pooled pairwise tests cannot be estimated with empty cells."
        )
    cell_mean, cell_n, cell_var = _cell_stats(data, dv, [f1, f2])
    dof = cell_n.sum() - len(cell_n)
    mse = np.nansum((cell_n - 1) * cell_var) / dof
    raw_groups = _group_arrays(data, [f1, f2], dv)
    rows = []
    # Main effects: estimated marginal means (equal-weight average of the cell means)
    for fac, other in ((f1, f2), (f2, f1)):
        n_other = data[other].nunique()
        emm = cell_mean.groupby(level=fac, observed=True, sort=True).mean()
        levels = list(emm.index)
        for a, b in combinations(levels, 2):
            n_a = cell_n.xs(a, level=fac)
            n_b = cell_n.xs(b, level=fac)
            # Variance of each EMM: variance of the average of the cell means across the levels
            # of the other factor, weighting each cell by its own sample size.
            se = np.sqrt(
                (mse / n_a).sum() / n_other**2 + (mse / n_b).sum() / n_other**2
            )
            # Effect size and (optionally) standard deviations use the raw observations of the
            # marginal groups, while the means are the estimated marginal means.
            x = data.loc[data[fac] == a, dv].to_numpy(dtype=np.float64)
            y = data.loc[data[fac] == b, dv].to_numpy(dtype=np.float64)
            desc = (
                (emm[a], np.nanstd(x, ddof=1), emm[b], np.nanstd(y, ddof=1))
                if return_desc
                else None
            )
            row = {"Contrast": fac, "A": a, "B": b, "Paired": False}
            row.update(
                _pooled_row(emm[a] - emm[b], se, dof, alternative, False, x, y, effsize, desc)
            )
            rows.append(row)
    # Interaction: cell means compared within each level of the first factor
    for fac1 in cell_mean.index.get_level_values(f1).unique():
        means = cell_mean.xs(fac1, level=f1)
        ns = cell_n.xs(fac1, level=f1)
        for a, b in combinations(means.index, 2):
            se = np.sqrt(mse * (1 / ns[a] + 1 / ns[b]))
            x, y = raw_groups[(fac1, a)], raw_groups[(fac1, b)]
            desc = (
                (means[a], np.nanstd(x, ddof=1), means[b], np.nanstd(y, ddof=1))
                if return_desc
                else None
            )
            row = {
                "Contrast": f"{f1} * {f2}",
                "Time": fac1,
                "A": a,
                "B": b,
                "Paired": False,
            }
            row.update(
                _pooled_row(means[a] - means[b], se, dof, alternative, False, x, y, effsize, desc)
            )
            rows.append(row)
    return rows


def _pooled_simple_within(data_piv, dv, within, effsize, alternative, return_desc):
    """Pooled comparisons for a single within-subject factor.

    The error term is the subject x condition interaction mean square of the one-way repeated
    measures ANOVA (:py:func:`pingouin.rm_anova`), with ``(n - 1)(k - 1)`` degrees of freedom.
    With only two levels the pooled T statistic and p-value are identical to the paired T-test.
    """
    n, k = data_piv.shape
    dof = (n - 1) * (k - 1)
    vals = data_piv.to_numpy(dtype=np.float64)
    cond_means = data_piv.mean(axis=0)
    subj_means = data_piv.mean(axis=1)
    grandmean = vals.mean()
    # SS of the subject x condition interaction (= residual of the repeated measures ANOVA)
    residual = (
        vals - subj_means.to_numpy()[:, None] - cond_means.to_numpy()[None, :] + grandmean
    )
    ss_error = (residual**2).sum()
    mse = ss_error / dof
    long = data_piv.stack(future_stack=True).rename(dv).reset_index()
    groups = _group_arrays(long, within, dv)
    rows = []
    for a, b in combinations(data_piv.columns, 2):
        x, y = groups[a], groups[b]
        se = np.sqrt(2 * mse / n)
        desc = (
            (cond_means[a], np.nanstd(x, ddof=1), cond_means[b], np.nanstd(y, ddof=1))
            if return_desc
            else None
        )
        row = {"Contrast": within, "A": a, "B": b, "Paired": True}
        row.update(
            _pooled_row(
                cond_means[a] - cond_means[b], se, dof, alternative, True, x, y, effsize, desc
            )
        )
        rows.append(row)
    return rows


def _mixed_error_terms(data_piv, long, dv, within, between):
    """Between- and within-subject error mean squares and degrees of freedom of a mixed ANOVA.

    This replicates the split-plot sums of squares computed by :py:func:`pingouin.mixed_anova`.
    """
    n_rm = data_piv.shape[1]
    n_subj = data_piv.shape[0]
    n_groups = long[between].nunique()
    grandmean = long[dv].mean(numeric_only=True)
    grp_b = long.groupby(between, observed=True)[dv]
    grp_w = long.groupby(within, observed=True)[dv]
    ss_betw = ((grp_b.mean(numeric_only=True) - grandmean) ** 2 * grp_b.count()).sum()
    ss_with = ((grp_w.mean(numeric_only=True) - grandmean) ** 2 * grp_w.count()).sum()
    grp = long.groupby([between, within], observed=True)[dv]
    ss_resall = ((long[dv] - grp.transform("mean")) ** 2).sum()
    ss_total = ((long[dv] - grandmean) ** 2).sum()
    ss_inter = ss_total - (ss_resall + ss_with + ss_betw)
    ss_subj = n_rm * ((data_piv.mean(axis=1) - grandmean) ** 2).sum()
    ss_resbetw = ss_subj - ss_betw
    ss_reswith = ss_total - (ss_with + ss_subj + ss_inter)
    df_resbetw = n_subj - n_groups
    df_reswith = (n_rm - 1) * df_resbetw
    return (
        ss_resbetw / df_resbetw,
        df_resbetw,
        ss_reswith / df_reswith,
        df_reswith,
        n_rm,
    )


def _pooled_mixed(
    data_piv, long, dv, within, between, within_first,
    effsize, alternative, return_desc,
):
    """Pooled comparisons for a mixed (one within x one between) design.

    The within-subject main effect and the interaction use the within-subject error term of
    :py:func:`pingouin.mixed_anova` (subject x within interaction), while the between-subject main
    effect uses the between-subject error term (subjects nested in groups). When the interaction
    compares the between-subject groups at each level of the within factor
    (``within_first=True``), the standard error of the cell-mean difference combines both error
    mean squares and its degrees of freedom use a Welch-Satterthwaite approximation.
    """
    ms_b, df_b, ms_w, df_w, n_rm = _mixed_error_terms(data_piv, long, dv, within, between)
    groups_w = _group_arrays(long, within, dv)
    groups_both = _group_arrays(long, [between, within], dv)
    n_subj = data_piv.shape[0]
    within_rows = []
    between_rows = []

    # Within-subject main effect: marginal means across all the subjects, within-subject error
    w_means = long.groupby(within, observed=True, sort=True)[dv].mean()
    for a, b in combinations(w_means.index, 2):
        x, y = groups_w[a], groups_w[b]
        se = np.sqrt(2 * ms_w / n_subj)
        desc = (
            (w_means[a], np.nanstd(x, ddof=1), w_means[b], np.nanstd(y, ddof=1))
            if return_desc
            else None
        )
        row = {"Contrast": within, "A": a, "B": b, "Paired": True}
        row.update(
            _pooled_row(w_means[a] - w_means[b], se, df_w, alternative, True, x, y, effsize, desc)
        )
        within_rows.append(row)

    # Between-subject main effect: one mean per subject, between-subject error term.
    # var(subject mean) = (MS_between_error + (k - 1) * MS_within_error) / k = MS_b / k, so the
    # SE uses MS_b / k.
    subj_mean = data_piv.mean(axis=1)
    b_means = subj_mean.groupby(level=between, observed=True, sort=True).mean()
    b_n = subj_mean.groupby(level=between, observed=True, sort=True).count()
    groups_b = _group_arrays(
        data_piv.mean(axis=1).rename(dv).reset_index(), between, dv
    )
    for a, b in combinations(b_means.index, 2):
        x, y = groups_b[a], groups_b[b]
        se = np.sqrt((ms_b / n_rm) * (1 / b_n[a] + 1 / b_n[b]))
        desc = (
            (b_means[a], np.nanstd(x, ddof=1), b_means[b], np.nanstd(y, ddof=1))
            if return_desc
            else None
        )
        row = {"Contrast": between, "A": a, "B": b, "Paired": False}
        row.update(
            _pooled_row(b_means[a] - b_means[b], se, df_b, alternative, False, x, y, effsize, desc)
        )
        between_rows.append(row)

    # The main effects follow the factor order: within first by default, between first otherwise.
    rows = within_rows + between_rows if within_first else between_rows + within_rows

    # Interaction
    cell_grp = long.groupby([between, within], observed=True, sort=True)[dv]
    cell_mean = cell_grp.mean()
    cell_n = cell_grp.count()
    if not within_first:
        # Within each group, compare the levels of the within factor: within-subject error.
        # Group labels follow the alphabetical (groupby) ordering, not their order of appearance
        # in the wide-format table.
        contrast = f"{between} * {within}"
        for grp in long.groupby(between, observed=True, sort=True).groups.keys():
            sub = data_piv.xs(grp, level=between)
            n_g = sub.shape[0]
            means = sub.mean(axis=0)
            for a, b in combinations(data_piv.columns, 2):
                x, y = groups_both[(grp, a)], groups_both[(grp, b)]
                se = np.sqrt(2 * ms_w / n_g)
                desc = (
                    (means[a], np.nanstd(x, ddof=1), means[b], np.nanstd(y, ddof=1))
                    if return_desc
                    else None
                )
                row = {
                    "Contrast": contrast,
                    "Time": grp,
                    "A": a,
                    "B": b,
                    "Paired": True,
                }
                row.update(
                    _pooled_row(
                        means[a] - means[b], se, df_w, alternative, True, x, y, effsize, desc
                    )
                )
                rows.append(row)
    else:
        # At each level of the within factor, compare the groups: cell means. Under the
        # split-plot model Y_sj = fixed effects + u_s + e_sj, the between-subject error MS on
        # the subject-mean scale is MS_b / k (so that E(MS_b / k) = sigma_u^2 + sigma_e^2 / k)
        # and E(MS_w) = sigma_e^2. The unbiased estimate of the variance of a cell mean over n
        # subjects is (MS_b / k) / n + MS_w * (k - 1) / (k n); the variance of the difference is
        # the sum of the two cell estimates. The degrees of freedom combine the two independent
        # mean squares with the Welch-Satterthwaite equation.
        ms_b_mean = ms_b / n_rm
        contrast = f"{within} * {between}"
        for tlevel in data_piv.columns:
            means = cell_mean.xs(tlevel, level=within)
            ns = cell_n.xs(tlevel, level=within)
            for a, b in combinations(means.index, 2):
                c1_a, c2_a = 1 / ns[a], (n_rm - 1) / (n_rm * ns[a])
                c1_b, c2_b = 1 / ns[b], (n_rm - 1) / (n_rm * ns[b])
                v_a = c1_a * ms_b_mean + c2_a * ms_w
                v_b = c1_b * ms_b_mean + c2_b * ms_w
                v = v_a + v_b
                se = np.sqrt(v)
                dof = v**2 / (
                    (c1_a + c1_b) ** 2 * ms_b_mean**2 / df_b
                    + (c2_a + c2_b) ** 2 * ms_w**2 / df_w
                )
                x, y = groups_both[(a, tlevel)], groups_both[(b, tlevel)]
                desc = (
                    (means[a], np.nanstd(x, ddof=1), means[b], np.nanstd(y, ddof=1))
                    if return_desc
                    else None
                )
                row = {
                    "Contrast": contrast,
                    "Time": tlevel,
                    "A": a,
                    "B": b,
                    "Paired": False,
                }
                row.update(
                    _pooled_row(
                        means[a] - means[b], se, dof, alternative, False, x, y, effsize, desc
                    )
                )
                rows.append(row)
    return rows


def _pooled_pairwise_tests(
    data, data_piv, dv, between, within, subject, contrast, factors,
    interaction, within_first, alternative, padjust, alpha, effsize, return_desc,
):
    """Run all the pooled pairwise tests, one family of contrasts per effect, and assemble them.

    Each contrast (factor main effect or interaction) is adjusted for multiple comparisons on its
    own, following the same family grouping as the non-pooled :py:func:`pairwise_tests`.
    """
    if contrast == "simple_between":
        rows = _pooled_simple_between(data, dv, between, effsize, alternative, return_desc)
        interaction_name = None
    elif contrast == "multiple_between":
        rows = _pooled_multiple_between(data, dv, factors, effsize, alternative, return_desc)
        interaction_name = f"{factors[0]} * {factors[1]}"
    elif contrast == "simple_within":
        rows = _pooled_simple_within(data_piv, dv, within, effsize, alternative, return_desc)
        interaction_name = None
    else:
        rows = _pooled_mixed(
            data_piv, data, dv, within, between, within_first,
            effsize, alternative, return_desc,
        )
        interaction_name = f"{within} * {between}" if within_first else f"{between} * {within}"

    if not interaction and interaction_name is not None:
        rows = [row for row in rows if row["Contrast"] != interaction_name]

    stats = []
    for name in dict.fromkeys(row["Contrast"] for row in rows):
        family = pd.DataFrame([row for row in rows if row["Contrast"] == name])
        stats.append(_padjust(family, padjust=padjust, alpha=alpha))

    stats = pd.concat(stats, axis=0, ignore_index=True, sort=False)
    stats["Parametric"] = True
    return stats


@pf.register_dataframe_method
def pairwise_ttests(*args, **kwargs):
    """This function has been deprecated . Use :py:func:`pingouin.pairwise_tests` instead."""
    warnings.warn("pairwise_ttests is deprecated, use pairwise_tests instead.", UserWarning)
    return pairwise_tests(*args, **kwargs)


@pf.register_dataframe_method
def pairwise_tests(
    data=None,
    dv=None,
    between=None,
    within=None,
    subject=None,
    parametric=True,
    marginal=True,
    alpha=0.05,
    alternative="two-sided",
    padjust="none",
    effsize="hedges",
    correction="auto",
    nan_policy="listwise",
    return_desc=False,
    interaction=True,
    within_first=True,
    pooled=False,
):
    """Pairwise tests.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    between : string or list with 2 elements
        Name of column(s) containing the between-subject factor(s).
    within : string or list with 2 elements
        Name of column(s) containing the within-subject factor(s), i.e. the
        repeated measurements. Only a single within-subject factor is supported for mixed analysis
        (if ``between`` is also specified).
    subject : string
        Name of column containing the subject identifier. This is mandatory when ``within`` is
        specified.
    parametric : boolean
        If True (default), use the parametric :py:func:`ttest` function.
        If False, use :py:func:`pingouin.wilcoxon` or :py:func:`pingouin.mwu`
        for paired or unpaired samples, respectively.
    marginal : boolean
        If True (default), the between-subject pairwise T-test(s) will be calculated
        after averaging across all levels of the within-subject factor in mixed
        design. This is recommended to avoid violating the assumption of
        independence and conflating the degrees of freedom by the
        number of repeated measurements.

        .. versionadded:: 0.3.2
    alpha : float
        Significance level
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return one-sided
        p-values. "greater" tests against the alternative hypothesis that the mean of ``x``
        is greater than the mean of ``y``.
    padjust : string
        Method used for testing and adjustment of pvalues.

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta_square'``: Eta-square
        * ``'odds_ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size
    correction : string or boolean
        For independent two sample T-tests, specify whether or not to correct for
        unequal variances using Welch separate variances T-test. If `'auto'`,
        it will automatically uses Welch T-test when the sample sizes are
        unequal, as recommended by Zimmerman 2004.

        .. versionadded:: 0.3.2
    nan_policy : string
        Can be `'listwise'` for listwise deletion of missing values in repeated
        measures design (= complete-case analysis) or `'pairwise'` for the
        more liberal pairwise deletion (= available-case analysis). The former (default) is more
        appropriate for post-hoc analysis following an ANOVA, however it can drastically reduce
        the power of the test: any subject with one or more missing value(s) will be
        completely removed from the analysis.

        .. versionadded:: 0.2.9
    return_desc : boolean
        If True, append group means and std to the output dataframe
    interaction : boolean
        If there are multiple factors and ``interaction`` is True (default),
        Pingouin will also calculate T-tests for the interaction term (see Notes).

        .. versionadded:: 0.2.9
    within_first : boolean
        Determines the order of the interaction in mixed design. Pingouin will
        return within * between when this parameter is set to True (default),
        and between * within otherwise.

        .. versionadded:: 0.3.6
    pooled : boolean
        If True, the parametric pairwise tests use the pooled error term of the corresponding
        ANOVA instead of estimating the variance separately from the two groups (or conditions)
        of each comparison. This makes the post-hoc tests consistent with the omnibus test: the
        T statistics are contrasts divided by an ANOVA-based standard error, and the degrees of
        freedom are those of the ANOVA error term. Only parametric tests are supported. A ``se``
        column with the standard error is added to the output, and no Bayes Factor is computed.

        The error term depends on the design:

        * **One between-subject factor**: the Within mean square of the one-way
          :py:func:`anova`, with ``N - k`` degrees of freedom. The standard errors are the
          Tukey-Kramer standard errors, so the T-values are identical to
          :py:func:`pairwise_tukey`.
        * **Two between-subject factors**: the residual mean square of the full factorial model
          (with the interaction), with ``N - number of cells`` degrees of freedom. The main
          effects compare estimated marginal means, i.e. the cell means averaged with equal
          weights across the levels of the other factor. When a factor has only two levels, the
          squared T of its contrast equals the F statistic of that factor in
          :py:func:`anova` (``ss_type=3``). The interaction contrasts compare cell means and
          have the same T-values as :py:func:`pairwise_tukey`.
        * **One within-subject factor**: the subject x condition mean square of
          :py:func:`rm_anova`, with ``(n - 1)(k - 1)`` degrees of freedom. With two levels the
          T and p-values are identical to the regular paired T-test.
        * **Mixed design (one within x one between)**: the within-subject main effect and the
          interaction use the within-subject error term of :py:func:`mixed_anova` (its ``DF2``),
          while the between-subject main effect, computed on each subject's mean, uses the
          between-subject error term (its ``DF2``). When the interaction compares the
          between-subject groups at each level of the within factor
          (``within_first=True``), the standard error combines both error mean squares of the
          split-plot model and its degrees of freedom follow a Welch-Satterthwaite
          approximation (they may be non-integer).

        Two within-subject factors are not supported. With two between-subject factors, all the
        cells (combinations of factor levels) must contain at least one observation.

        .. versionadded:: 0.7.1

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'Contrast'``: Contrast (= independent variable or interaction)
        * ``'A'``: Name of first measurement
        * ``'B'``: Name of second measurement
        * ``'Paired'``: indicates whether the two measurements are paired or
          independent
        * ``'Parametric'``: indicates if (non)-parametric tests were used
        * ``'T'``: T statistic (only if parametric=True)
        * ``'U_val'``: Mann-Whitney U stat (if parametric=False and unpaired
          data)
        * ``'W_val'``: Wilcoxon W stat (if parametric=False and paired data)
        * ``'dof'``: degrees of freedom (only if parametric=True)
        * ``'se'``: standard error of the estimated difference (only if ``pooled=True``)
        * ``'alternative'``: tail of the test
        * ``'p_unc'``: Uncorrected p-values
        * ``'p_corr'``: Corrected p-values
        * ``'p_adjust'``: p-values correction method
        * ``'BF10'``: Bayes Factor (not computed when ``pooled=True``)
        * ``'hedges'``: effect size (or any effect size defined in
          ``effsize``)

    See also
    --------
    ttest, mwu, wilcoxon, compute_effsize, multicomp

    Notes
    -----
    Data are expected to be in long-format. If your data is in wide-format,
    you can use the :py:func:`pandas.melt` function to convert from wide to
    long format.

    If ``between`` or ``within`` is a list (e.g. ['col1', 'col2']),
    the function returns 1) the pairwise T-tests between each values of the
    first column, 2) the pairwise T-tests between each values of the second
    column and 3) the interaction between col1 and col2. The interaction is
    dependent of the order of the list, so ['col1', 'col2'] will not yield the
    same results as ['col2', 'col1']. Furthermore, the interaction will only be
    calculated if ``interaction=True``.

    If ``between`` is a list with two elements, the output
    model is between1 + between2 + between1 * between2.

    Similarly, if ``within`` is a list with two elements, the output model is
    within1 + within2 + within1 * within2.

    If both ``between`` and ``within`` are specified, the output model is
    within + between + within * between (= mixed design), unless
    ``within_first=False`` in which case the model becomes between + within +
    between * within. Mixed analysis with two or more within-subject factors are not currently
    supported.

    Missing values in repeated measurements are automatically removed using a
    listwise (default) or pairwise deletion strategy. The former is more conservative, as any
    subject with one or more missing value(s) will be completely removed from the dataframe prior
    to calculating the T-tests. The ``nan_policy`` parameter can therefore have a huge impact
    on the results.

    Examples
    --------
    For more examples, please refer to the `Jupyter notebooks
    <https://github.com/raphaelvallat/pingouin/blob/master/notebooks/01_ANOVA.ipynb>`_

    1. One between-subject factor

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> pd.set_option("display.max_columns", 20)
    >>> df = pg.read_dataset("mixed_anova.csv")
    >>> pg.pairwise_tests(dv="Scores", between="Group", data=df).round(3)
      Contrast        A           B  Paired  Parametric     T    dof alternative  p_unc   BF10  hedges
    0    Group  Control  Meditation   False        True -2.29  178.0   two-sided  0.023  1.813   -0.34

    2. One within-subject factor

    >>> post_hocs = pg.pairwise_tests(dv="Scores", within="Time", subject="Subject", data=df)
    >>> post_hocs.round(3)
      Contrast        A        B  Paired  Parametric      T   dof alternative  p_unc   BF10  hedges
    0     Time   August  January    True        True -1.740  59.0   two-sided  0.087  0.582  -0.328
    1     Time   August     June    True        True -2.743  59.0   two-sided  0.008  4.232  -0.483
    2     Time  January     June    True        True -1.024  59.0   two-sided  0.310  0.232  -0.170

    3. Non-parametric pairwise paired test (wilcoxon)

    >>> pg.pairwise_tests(
    ...     dv="Scores", within="Time", subject="Subject", data=df, parametric=False
    ... ).round(3)
      Contrast        A        B  Paired  Parametric  W_val alternative  p_unc  hedges
    0     Time   August  January    True       False  716.0   two-sided  0.144  -0.328
    1     Time   August     June    True       False  564.0   two-sided  0.010  -0.483
    2     Time  January     June    True       False  887.0   two-sided  0.840  -0.170

    4. Mixed design (within and between) with bonferroni-corrected p-values

    >>> posthocs = pg.pairwise_tests(
    ...     dv="Scores", within="Time", subject="Subject", between="Group", padjust="bonf", data=df
    ... )
    >>> posthocs.round(3)
           Contrast     Time        A           B Paired  Parametric      T   dof alternative  p_unc  p_corr p_adjust   BF10  hedges
    0          Time        -   August     January   True        True -1.740  59.0   two-sided  0.087   0.261     bonf  0.582  -0.328
    1          Time        -   August        June   True        True -2.743  59.0   two-sided  0.008   0.024     bonf  4.232  -0.483
    2          Time        -  January        June   True        True -1.024  59.0   two-sided  0.310   0.931     bonf  0.232  -0.170
    3         Group        -  Control  Meditation  False        True -2.248  58.0   two-sided  0.028     NaN      NaN  2.096  -0.573
    4  Time * Group   August  Control  Meditation  False        True  0.316  58.0   two-sided  0.753   1.000     bonf  0.274   0.081
    5  Time * Group  January  Control  Meditation  False        True -1.434  58.0   two-sided  0.157   0.471     bonf  0.619  -0.365
    6  Time * Group     June  Control  Meditation  False        True -2.744  58.0   two-sided  0.008   0.024     bonf  5.593  -0.699

    5. Two between-subject factors. The order of the ``between`` factors matters!

    >>> pg.pairwise_tests(dv="Scores", between=["Group", "Time"], data=df).round(3)
           Contrast       Group        A           B Paired  Parametric      T    dof alternative  p_unc     BF10  hedges
    0         Group           -  Control  Meditation  False        True -2.290  178.0   two-sided  0.023    1.813  -0.340
    1          Time           -   August     January  False        True -1.806  118.0   two-sided  0.074    0.839  -0.328
    2          Time           -   August        June  False        True -2.660  118.0   two-sided  0.009    4.499  -0.483
    3          Time           -  January        June  False        True -0.934  118.0   two-sided  0.352    0.288  -0.170
    4  Group * Time     Control   August     January  False        True -0.383   58.0   two-sided  0.703    0.279  -0.098
    5  Group * Time     Control   August        June  False        True -0.292   58.0   two-sided  0.771    0.272  -0.074
    6  Group * Time     Control  January        June  False        True  0.045   58.0   two-sided  0.964    0.263   0.011
    7  Group * Time  Meditation   August     January  False        True -2.188   58.0   two-sided  0.033    1.884  -0.558
    8  Group * Time  Meditation   August        June  False        True -4.040   58.0   two-sided  0.000  148.302  -1.030
    9  Group * Time  Meditation  January        June  False        True -1.442   58.0   two-sided  0.155    0.625  -0.367

    6. Same but without the interaction, and using a directional test

    >>> df.pairwise_tests(
    ...     dv="Scores", between=["Group", "Time"], alternative="less", interaction=False
    ... ).round(3)
      Contrast        A           B  Paired  Parametric      T    dof alternative  p_unc  hedges
    0    Group  Control  Meditation   False        True -2.290  178.0        less  0.012  -0.340
    1     Time   August     January   False        True -1.806  118.0        less  0.037  -0.328
    2     Time   August        June   False        True -2.660  118.0        less  0.004  -0.483
    3     Time  January        June   False        True -0.934  118.0        less  0.176  -0.170

    7. Pooled error term, consistent with the omnibus ANOVA. The standard error (``se``) is
    reported but no Bayes Factor is computed.

    >>> pg.pairwise_tests(dv="Scores", between="Group", data=df, pooled=True).round(3)
      Contrast        A           B  Paired  Parametric     T    dof     se alternative  p_unc  hedges
    0    Group  Control  Meditation   False        True -2.29  178.0  0.152   two-sided  0.023  -0.34
    """
    # Safety checks
    data = _check_dataframe(
        dv=dv, between=between, within=within, subject=subject, effects="all", data=data
    )
    _check_alternative(alternative)
    assert isinstance(alpha, float), "alpha must be float."
    assert nan_policy in ["listwise", "pairwise"]
    if not isinstance(pooled, bool):
        raise ValueError("pooled must be a boolean.")
    if pooled:
        if not parametric:
            raise ValueError("Pooled pairwise tests are only available for parametric tests.")
        if nan_policy == "pairwise":
            raise ValueError(
                "Pooled pairwise tests require listwise deletion of missing values "
                "(nan_policy='listwise')."
            )

    # Check if we have multiple between or within factors
    for name, factor in [("between", between), ("within", within)]:
        if isinstance(factor, list) and len(factor) > 2:
            raise ValueError(
                f"At most two {name} factors are supported, got {len(factor)}: {factor}."
            )
    contrast = None
    if isinstance(between, list):
        if len(between) > 1:
            contrast = "multiple_between"
            assert all(b in data.keys() for b in between)
        else:
            between = between[0]
    if isinstance(within, list):
        if len(within) > 1:
            if contrast == "multiple_between":
                raise ValueError(
                    "Multiple between and within factors are currently not supported. "
                    "Please select only one."
                )
            contrast = "multiple_within"
            assert all(w in data.keys() for w in within)
        else:
            within = within[0]
    # Check the other cases. Between and within column names can be str or int (not float).
    if isinstance(between, (str, int)) and within is None:
        contrast = "simple_between"
        assert between in data.keys()
    if isinstance(within, (str, int)) and between is None:
        contrast = "simple_within"
        assert within in data.keys()
    if isinstance(between, (str, int)) and isinstance(within, (str, int)):
        contrast = "within_between"
        assert between in data.keys() and within in data.keys()

    if pooled:
        if contrast == "multiple_within":
            raise ValueError(
                "Pooled pairwise tests are not supported for two within-subject factors."
            )
        if contrast == "within_between" and not marginal:
            raise ValueError(
                "Pooled pairwise tests in a mixed design require marginal=True for the "
                "between-subject factor."
            )

    # Create col_order. The standard error is only reported for pooled parametric tests, and the
    # Bayes Factor is not computed in that case.
    col_order = [
        "Contrast",
        "Time",
        "A",
        "B",
        "mean_A",
        "std_A",
        "mean_B",
        "std_B",
        "Paired",
        "Parametric",
        "T",
        "U_val",
        "W_val",
        "dof",
        "se",
        "alternative",
        "p_unc",
        "p_corr",
        "p_adjust",
        "BF10",
        effsize,
    ]
    if pooled:
        col_order.remove("BF10")

    # If repeated measures, pivot and stack the table. This has several effects:
    # 1) Force missing values to be explicit (a NaN cell is created)
    # 2) Automatic collapsing to the mean if multiple within factors are present
    # 3) If using dropna, remove rows with missing values (listwise deletion).
    # The latter is the same behavior as JASP (= strict complete-case analysis).
    if within is not None:
        idx_piv = subject if between is None else [subject, between]
        data_piv = data.pivot_table(index=idx_piv, columns=within, values=dv, observed=True)
        if nan_policy == "listwise":
            # Remove rows (= subject) with missing values. For pairwise deletion, missing values
            # will be removed directly in the lower-level functions (e.g. pg.ttest)
            data_piv = data_piv.dropna()
        # Stack instead of melt, which fails when a level of the within factor(s) is named as dv
        data = data_piv.stack(list(range(data_piv.columns.nlevels)), future_stack=True)
        data = data.rename(dv).reset_index()
    else:
        data_piv = None

    if pooled:
        # Pooled error terms: each contrast uses the error term of the matching ANOVA, so the
        # post-hoc T statistics are consistent with the omnibus F tests.
        if contrast in ("simple_between", "multiple_between"):
            factors = [between] if contrast == "simple_between" else between
        elif contrast == "simple_within":
            factors = [within]
        else:
            factors = [within, between] if within_first else [between, within]
        with _no_rounding():  # For max precision
            stats = _pooled_pairwise_tests(
                data=data,
                data_piv=data_piv,
                dv=dv,
                between=between,
                within=within,
                subject=subject,
                contrast=contrast,
                factors=factors,
                interaction=interaction,
                within_first=within_first,
                alternative=alternative,
                padjust=padjust,
                alpha=alpha,
                effsize=effsize,
                return_desc=return_desc,
            )
        # Reorder and drop empty columns
        stats = stats[[c for c in col_order if c in stats.columns]]
        stats = stats.dropna(how="all", axis=1)

        # Rename Time columns
        if "Time" in stats.columns:
            stats["Time"] = stats["Time"].fillna("-")
            stats = stats.rename(columns={"Time": factors[0]})

        return _postprocess_dataframe(stats)

    # Define the main effects, as (factor, paired, marginal) tuples, and the interaction.
    # Within-subject factors are always tested on the marginal means of each subject, i.e. after
    # averaging across the levels of the other within-subject factor. In mixed design, the
    # between-subject factor is tested on the marginal means if ``marginal=True``, to avoid
    # violating the assumption of independence and conflating the degrees of freedom by the
    # number of repeated measurements.
    if contrast == "simple_between":
        factors = [between]
        main_effects = [(between, False, False)]
    elif contrast == "simple_within":
        factors = [within]
        main_effects = [(within, True, False)]
    elif contrast == "multiple_between":
        # BETWEEN1 + BETWEEN2 + BETWEEN1 * BETWEEN2
        factors = between
        main_effects = [(f, False, False) for f in factors]
        paired_interaction = False
    elif contrast == "multiple_within":
        # WITHIN1 + WITHIN2 + WITHIN1 * WITHIN2
        factors = within
        main_effects = [(f, True, True) for f in factors]
        paired_interaction = True
    elif within_first:
        # within + between + within * between
        factors = [within, between]
        main_effects = [(within, True, True), (between, False, marginal)]
        paired_interaction = False  # Groups are compared at each level of the within factor
    else:
        # between + within + between * within
        factors = [between, within]
        main_effects = [(between, False, marginal), (within, True, True)]
        paired_interaction = True  # Levels are compared within each group

    # Keyword arguments shared by all the pairwise tests
    test_kwargs = dict(
        parametric=parametric,
        alternative=alternative,
        correction=correction,
        effsize=effsize,
        return_desc=return_desc,
    )

    stats = []
    with _no_rounding():  # For max precision
        # Main effects: one family of tests (for p-values correction) per factor
        for f, paired, agg in main_effects:
            tmp = data
            if agg:
                tmp = data.groupby([subject, f], as_index=False, observed=True, sort=True).mean(
                    numeric_only=True
                )
            # Extract levels of the grouping variable, sorted in alphabetical order
            groups = _group_arrays(tmp, f, dv)
            if len(groups) < 2:
                raise ValueError("Columns must have at least two unique values.")
            rows = []
            for a, b in combinations(groups, 2):
                row = {"Contrast": f, "A": a, "B": b, "Paired": paired}
                row.update(_pairwise_test(groups[a], groups[b], paired=paired, **test_kwargs))
                rows.append(row)
            stats.append(_padjust(pd.DataFrame(rows), padjust=padjust, alpha=alpha))

        # Interaction: compare the levels of the second factor at each level of the first factor
        if interaction and len(factors) == 2:
            # BUGFIX 0.3.9: If subject is present, make sure that we respect
            # the order of subjects.
            if subject is not None:
                data = data.set_index(subject).sort_index()
            # Extract interaction levels, sorted in alphabetical order
            labels_fac1 = data.groupby(factors[0], observed=True, sort=True).groups.keys()
            labels_fac2 = data.groupby(factors[1], observed=True, sort=True).groups.keys()
            groups = _group_arrays(data, factors, dv)
            rows = []
            for fac1, (a, b) in product(labels_fac1, combinations(labels_fac2, 2)):
                row = {
                    "Contrast": f"{factors[0]} * {factors[1]}",
                    "Time": fac1,
                    "A": a,
                    "B": b,
                    "Paired": paired_interaction,
                }
                x, y = groups[(fac1, a)], groups[(fac1, b)]
                row.update(_pairwise_test(x, y, paired=paired_interaction, **test_kwargs))
                rows.append(row)
            stats.append(_padjust(pd.DataFrame(rows), padjust=padjust, alpha=alpha))

    stats = pd.concat(stats, axis=0, ignore_index=True, sort=False)
    stats["Parametric"] = parametric

    # Reorder and drop empty columns
    stats = stats[[c for c in col_order if c in stats.columns]]
    stats = stats.dropna(how="all", axis=1)

    # Rename Time columns
    if "Time" in stats.columns:
        stats["Time"] = stats["Time"].fillna("-")
        stats = stats.rename(columns={"Time": factors[0]})

    return _postprocess_dataframe(stats)


@pf.register_dataframe_method
def ptests(
    self,
    paired=False,
    decimals=3,
    padjust=None,
    stars=True,
    pval_stars={0.001: "***", 0.01: "**", 0.05: "*"},
    **kwargs,
):
    """
    Pairwise T-test between columns of a dataframe.

    T-values are reported on the lower triangle of the output pairwise matrix and p-values on the
    upper triangle. This method is a faster, but less exhaustive, matrix-version of the
    :py:func:`pingouin.pairwise_test` function. Missing values are automatically removed from each
    pairwise T-test.

    .. versionadded:: 0.5.3

    Parameters
    ----------
    self : :py:class:`pandas.DataFrame`
        Input dataframe.
    paired : boolean
        Specify whether the two observations are related (i.e. repeated measures) or independent.
    decimals : int
        Number of decimals to display in the output matrix.
    padjust : string or None
        P-values adjustment for multiple comparison

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    stars : boolean
        If True, only significant p-values are displayed as stars using the pre-defined thresholds
        of ``pval_stars``. If False, all the raw p-values are displayed.
    pval_stars : dict
        Significance thresholds. Default is 3 stars for p-values <0.001, 2 stars for
        p-values <0.01 and 1 star for p-values <0.05.
    **kwargs : optional
        Optional argument(s) passed to the lower-level scipy functions, i.e.
        :py:func:`scipy.stats.ttest_ind` for independent T-test and
        :py:func:`scipy.stats.ttest_rel` for paired T-test.

    Returns
    -------
    mat : :py:class:`pandas.DataFrame`
        Pairwise T-test matrix, of dtype str, with T-values on the lower triangle and p-values on
        the upper triangle.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> # Load an example dataset of personality dimensions
    >>> df = pg.read_dataset("pairwise_corr").iloc[:30, 1:]
    >>> df.columns = ["N", "E", "O", "A", "C"]
    >>> # Add some missing values
    >>> df.iloc[[2, 5, 20], 2] = np.nan
    >>> df.iloc[[1, 4, 10], 3] = np.nan
    >>> df.head().round(2)
          N     E     O     A     C
    0  2.48  4.21  3.94  3.96  3.46
    1  2.60  3.19  3.96   NaN  3.23
    2  2.81  2.90   NaN  2.75  3.50
    3  2.90  3.56  3.52  3.17  2.79
    4  3.02  3.33  4.02   NaN  2.85

    Independent pairwise T-tests

    >>> df.ptests()
            N       E      O      A    C
    N       -     ***    ***    ***  ***
    E  -8.397       -                ***
    O  -8.332  -0.596      -         ***
    A  -8.804    0.12   0.72      -  ***
    C  -4.759   3.753  4.074  3.787    -

    Let's compare with SciPy

    >>> from scipy.stats import ttest_ind
    >>> np.round(ttest_ind(df["N"], df["E"]), 3)
    array([-8.397,  0.   ])

    Passing custom parameters to the lower-level :py:func:`scipy.stats.ttest_ind` function

    >>> df.ptests(alternative="greater", equal_var=True)
            N       E      O      A    C
    N       -
    E  -8.397       -                ***
    O  -8.332  -0.596      -         ***
    A  -8.804    0.12   0.72      -  ***
    C  -4.759   3.753  4.074  3.787    -

    Paired T-test, showing the actual p-values instead of stars

    >>> df.ptests(paired=True, stars=False, decimals=4)
            N        E       O       A       C
    N        -   0.0000  0.0000  0.0000  0.0002
    E  -7.0773        -  0.8776  0.7522  0.0012
    O  -8.0568  -0.1555       -  0.8137  0.0008
    A  -8.3994   0.3191  0.2383       -  0.0009
    C  -4.2511   3.5953  3.7849  3.7652       -

    Adjusting for multiple comparisons using the Holm-Bonferroni method

    >>> df.ptests(paired=True, stars=False, padjust="holm")
            N       E      O      A      C
    N       -   0.000  0.000  0.000  0.001
    E  -7.077       -     1.     1.  0.005
    O  -8.057  -0.155      -     1.  0.005
    A  -8.399   0.319  0.238      -  0.005
    C  -4.251   3.595  3.785  3.765      -
    """
    from scipy.stats import ttest_ind, ttest_rel

    assert isinstance(pval_stars, dict), "pval_stars must be a dictionary."
    assert isinstance(decimals, int), "decimals must be an int."
    if "axis" in kwargs or "nan_policy" in kwargs:
        raise ValueError(
            "axis and nan_policy cannot be specified: the T-tests are always calculated between "
            "the columns of the dataframe, with pairwise deletion of missing values."
        )

    func = ttest_rel if paired else ttest_ind

    # Get T-values and p-values, with all the pairs tested at once in a single vectorized call.
    # We cannot use pandas.DataFrame.corr here because it will incorrectly remove rows missing
    # values, even when using an independent T-test!
    cols = self.columns
    arr = self.to_numpy(dtype=np.float64)
    i1, i2 = np.triu_indices(len(cols), k=1)
    t, p = func(arr[:, i1], arr[:, i2], axis=0, nan_policy="omit", **kwargs)
    tvals = np.full((len(cols), len(cols)), np.nan)
    pvals = tvals.copy()
    tvals[i2, i1] = np.round(t, decimals)
    # Do not round p-value here, or we'll lose precision for multicomp
    pvals[i1, i2] = p
    mat = pd.DataFrame(tvals, columns=cols, index=cols)
    mat_upper = pd.DataFrame(pvals, columns=cols, index=cols)

    if padjust is not None:
        # Only the unique pairs (strict upper triangle) belong to the test family.
        mat_upper = _multicomp_triu(mat_upper, method=padjust, alpha=0.05)

    # T-values on the lower triangle and p-values (as stars or str) on the upper triangle
    return _format_pairwise_matrix(mat, mat_upper, decimals, stars, pval_stars=pval_stars)


@pf.register_dataframe_method
def pairwise_tukey(data=None, dv=None, between=None, effsize="hedges"):
    """Pairwise Tukey-HSD post-hoc test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a Pandas method, in which
        case this argument is no longer needed.
    dv : string
        Name of column containing the dependent variable.
    between : string or list with N elements
        Name of column(s) containing the between factor(s). If ``between`` is a list with two or
        more elements, all the combinations of levels (cells) of the factors are compared, i.e.
        the interaction term.
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta_square'``: Eta-square
        * ``'odds_ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'A'``: Name of first measurement (a tuple with one level per factor if ``between``
          has several factors)
        * ``'B'``: Name of second measurement
        * ``'mean_A'``: Mean of first measurement
        * ``'mean_B'``: Mean of second measurement
        * ``'diff'``: Mean difference (= mean(A) - mean(B))
        * ``'se'``: Standard error
        * ``'T'``: T-values
        * ``'p_tukey'``: Tukey-HSD corrected p-values
        * ``'hedges'``: Hedges effect size (or any effect size defined in
          ``effsize``)

    See also
    --------
    pairwise_tests, pairwise_gameshowell

    Notes
    -----
    Tukey HSD post-hoc [1]_ is best for balanced one-way ANOVA.

    It has been proven to be conservative for one-way ANOVA with unequal sample sizes. However, it
    is not robust if the groups have unequal variances, in which case the Games-Howell test is
    more adequate. Tukey HSD is not valid for repeated measures ANOVA.

    With several between factors, the groups are all the observed combinations of levels of the
    factors, and :math:`\\text{MS}_w` is the residual of the full factorial ANOVA (with all the
    interactions). This is the same as ``TukeyHSD(aov(dv ~ A * B), which = "A:B")`` in R.

    The T-values are defined as:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}
        {\\sqrt{2 \\cdot \\text{MS}_w / n}}

    where :math:`\\overline{x}_i` and :math:`\\overline{x}_j` are the means of the first and
    second group, respectively, :math:`\\text{MS}_w` the mean squares of the error (computed using
    ANOVA) and :math:`n` the sample size.

    If the sample sizes are unequal, the Tukey-Kramer procedure is automatically used:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}{\\sqrt{\\frac{MS_w}{n_i}
        + \\frac{\\text{MS}_w}{n_j}}}

    where :math:`n_i` and :math:`n_j` are the sample sizes of the first and second group,
    respectively.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\\sqrt2|t_i|, r, N - r)` where :math:`r` is the total number of groups and
    :math:`N` is the total sample size.

    References
    ----------
    .. [1] Tukey, John W. "Comparing individual means in the analysis of
           variance." Biometrics (1949): 99-114.

    .. [2] Gleason, John R. "An accurate, non-iterative approximation for
           studentized range quantiles." Computational statistics & data
           analysis 31.2 (1999): 147-158.

    Examples
    --------
    Pairwise Tukey post-hocs on the Penguins dataset.

    >>> import pingouin as pg
    >>> df = pg.read_dataset("penguins")
    >>> df.pairwise_tukey(dv="body_mass_g", between="species").round(3)
               A          B    mean_A    mean_B      diff      se       T  p_tukey  hedges
    0     Adelie  Chinstrap  3700.662  3733.088   -32.426  67.512  -0.480    0.881  -0.074
    1     Adelie     Gentoo  3700.662  5076.016 -1375.354  56.148 -24.495    0.000  -2.860
    2  Chinstrap     Gentoo  3733.088  5076.016 -1342.928  69.857 -19.224    0.000  -2.875

    Tukey post-hocs on the interaction of species and sex, i.e. all the pairs of species x sex
    combinations. This is the same as ``TukeyHSD(aov(body_mass_g ~ species * sex),
    which = "species:sex")`` in R.

    >>> stats = pg.pairwise_tukey(data=df, dv="body_mass_g", between=["species", "sex"])
    >>> stats[["A", "B", "diff", "T", "p_tukey"]].head(4).round(3)
                      A                    B      diff       T  p_tukey
    0  (Adelie, female)       (Adelie, male)  -674.658 -13.174    0.000
    1  (Adelie, female)  (Chinstrap, female)  -158.370  -2.465    0.138
    2  (Adelie, female)    (Chinstrap, male)  -570.135  -8.875    0.000
    3  (Adelie, female)     (Gentoo, female) -1310.906 -24.088    0.000
    """
    data = _check_dataframe(dv=dv, between=between, effects="between", data=data)
    stats, n, gvars = _pairwise_between(data, dv, between, effsize)
    ng = n.size
    g1, g2 = np.triu_indices(ng, k=1)

    # Mean squares of the error, pooled across all the groups (or cells). This is the residual of
    # the one-way ANOVA, or of the full factorial ANOVA when there are several factors.
    # Groups with a single observation have an undefined variance but no residual.
    df = n.sum() - ng
    ms_within = np.nansum((n - 1) * gvars) / df
    gvar = ms_within / n
    stats["se"] = np.sqrt(gvar[g1] + gvar[g2])
    stats["T"] = stats["diff"] / stats["se"]

    # P-values
    pval = studentized_range.sf(np.sqrt(2) * np.abs(stats["T"]), ng, df)
    stats["p_tukey"] = np.clip(pval, 0, 1)
    return _postprocess_dataframe(stats[[*stats.columns.drop(effsize), effsize]])


@pf.register_dataframe_method
def pairwise_gameshowell(data=None, dv=None, between=None, effsize="hedges"):
    """Pairwise Games-Howell post-hoc test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame
    dv : string
        Name of column containing the dependent variable.
    between : string or list with N elements
        Name of column(s) containing the between factor(s). If ``between`` is a list with two or
        more elements, all the combinations of levels (cells) of the factors are compared, i.e.
        the interaction term.
    effsize : string or None
        Effect size type. Available methods are:

        * ``'none'``: no effect size
        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'r'``: Pearson correlation coefficient
        * ``'eta_square'``: Eta-square
        * ``'odds_ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'CLES'``: Common Language Effect Size

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`
        Stats summary:

        * ``'A'``: Name of first measurement (a tuple with one level per factor if ``between``
          has several factors)
        * ``'B'``: Name of second measurement
        * ``'mean_A'``: Mean of first measurement
        * ``'mean_B'``: Mean of second measurement
        * ``'diff'``: Mean difference (= mean(A) - mean(B))
        * ``'se'``: Standard error
        * ``'T'``: T-values
        * ``'df'``: adjusted degrees of freedom
        * ``'pval'``: Games-Howell corrected p-values
        * ``'hedges'``: Hedges effect size (or any effect size defined in
          ``effsize``)

    See also
    --------
    pairwise_tests, pairwise_tukey

    Notes
    -----
    Games-Howell [1]_ is very similar to the Tukey HSD post-hoc test but is much more robust to
    heterogeneity of variances. While the Tukey-HSD post-hoc is optimal after a classic one-way
    ANOVA, the Games-Howell is optimal after a Welch ANOVA. Please note that Games-Howell
    is not valid for repeated measures ANOVA. With several between factors, the groups are all
    the observed combinations of levels of the factors.

    Compared to the Tukey-HSD test, the Games-Howell test uses different pooled variances for
    each pair of variables instead of the same pooled variance.

    The T-values are defined as:

    .. math::

        t = \\frac{\\overline{x}_i - \\overline{x}_j}
        {\\sqrt{(\\frac{s_i^2}{n_i} + \\frac{s_j^2}{n_j})}}

    and the corrected degrees of freedom are:

    .. math::

        v = \\frac{(\\frac{s_i^2}{n_i} + \\frac{s_j^2}{n_j})^2}
        {\\frac{(\\frac{s_i^2}{n_i})^2}{n_i-1} +
        \\frac{(\\frac{s_j^2}{n_j})^2}{n_j-1}}

    where :math:`\\overline{x}_i`, :math:`s_i^2`, and :math:`n_i` are the mean, variance and sample
    size of the first group and :math:`\\overline{x}_j`, :math:`s_j^2`, and :math:`n_j` the mean,
    variance and sample size of the second group.

    The p-values are then approximated using the Studentized range distribution
    :math:`Q(\\sqrt2|t_i|, r, v_i)`.

    References
    ----------
    .. [1] Games, Paul A., and John F. Howell. "Pairwise multiple comparison
           procedures with unequal n's and/or variances: a Monte Carlo study."
           Journal of Educational Statistics 1.2 (1976): 113-125.

    .. [2] Gleason, John R. "An accurate, non-iterative approximation for
           studentized range quantiles." Computational statistics & data
           analysis 31.2 (1999): 147-158.

    Examples
    --------
    Pairwise Games-Howell post-hocs on the Penguins dataset.

    >>> import pingouin as pg
    >>> df = pg.read_dataset("penguins")
    >>> pg.pairwise_gameshowell(data=df, dv="body_mass_g", between="species").round(3)
               A          B    mean_A    mean_B      diff      se       T       df  pval  hedges
    0     Adelie  Chinstrap  3700.662  3733.088   -32.426  59.706  -0.543  152.455  0.85  -0.074
    1     Adelie     Gentoo  3700.662  5076.016 -1375.354  58.811 -23.386  249.643  0.00  -2.860
    2  Chinstrap     Gentoo  3733.088  5076.016 -1342.928  65.103 -20.628  170.404  0.00  -2.875
    """
    # Check the dataframe
    data = _check_dataframe(dv=dv, between=between, effects="between", data=data)
    stats, n, gvars = _pairwise_between(data, dv, between, effsize)
    ng = n.size
    g1, g2 = np.triu_indices(ng, k=1)

    # Standard errors and Welch-Satterthwaite degrees of freedom, using unpooled variances
    v = gvars / n
    stats["se"] = np.sqrt(v[g1] + v[g2])
    stats["T"] = stats["diff"] / stats["se"]
    stats["df"] = (v[g1] + v[g2]) ** 2 / (v[g1] ** 2 / (n[g1] - 1) + v[g2] ** 2 / (n[g2] - 1))

    # Compute corrected p-values
    pval = studentized_range.sf(np.sqrt(2) * np.abs(stats["T"]), ng, stats["df"])
    stats["pval"] = np.clip(pval, 0, 1)
    return _postprocess_dataframe(stats[[*stats.columns.drop(effsize), effsize]])


def _pairwise_between(data, dv, between, effsize):
    """Group summary and exact effect sizes of all the pairs of levels of the between factor(s).

    Shared by :py:func:`pairwise_tukey` and :py:func:`pairwise_gameshowell`. Returns the output
    dataframe (A, B, mean_A, mean_B, diff and effect size columns, one row per pair), and the
    per-group sample sizes and variances.
    """
    between = _flatten_list([between])
    data = data[[dv, *between]].dropna()
    if len(between) == 1:
        between = between[0]  # Scalar labels instead of 1-tuples
    # Levels are sorted with groupby (default sort=True) and not np.unique, because the latter
    # does not respect the custom sorting of a Categorical column.
    # See https://github.com/raphaelvallat/pingouin/issues/111
    # With several factors, the groups are all the observed combinations of levels (cells), and
    # the labels are tuples with one level per factor.
    grp = data.groupby(between, observed=True)[dv]
    groups = {k: v.to_numpy(dtype=np.float64) for k, v in grp}
    labels = np.empty(len(groups), dtype=object)
    labels[:] = list(groups)
    n = grp.count().to_numpy()
    gmeans = grp.mean(numeric_only=True).to_numpy()
    gvars = grp.var(numeric_only=True).to_numpy()
    g1, g2 = np.triu_indices(len(labels), k=1)
    stats = pd.DataFrame(
        {
            "A": labels[g1],
            "B": labels[g2],
            "mean_A": gmeans[g1],
            "mean_B": gmeans[g2],
            "diff": gmeans[g1] - gmeans[g2],
            effsize: [
                compute_effsize(groups[labels[a]], groups[labels[b]], paired=False, eftype=effsize)
                for a, b in zip(g1, g2)
            ],
        }
    )
    return stats, n, gvars


@pf.register_dataframe_method
def pairwise_corr(
    data,
    columns=None,
    covar=None,
    alternative="two-sided",
    method="pearson",
    padjust="none",
    nan_policy="pairwise",
):
    """Pairwise (partial) correlations between columns of a pandas dataframe.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        DataFrame. Note that this function can also directly be used as a
        Pandas method, in which case this argument is no longer needed.
    columns : list or str
        Column names in data:

        * ``["a", "b", "c"]``: combination between columns a, b, and c.
        * ``["a"]``: product between a and all the other numeric columns.
        * ``[["a"], ["b", "c"]]``: product between ["a"] and ["b", "c"].
        * ``[["a", "d"], ["b", "c"]]``: product between ["a", "d"] and
          ["b", "c"].
        * ``[["a", "d"], None]``: product between ["a", "d"] and all other
          numeric columns in dataframe.

        If column is None, the function will return the pairwise correlation
        between the combination of all the numeric columns in data.
        See the examples section for more details on this.
    covar : None, string or list
        Covariate(s) for partial correlation. Must be one or more columns
        in data. Use a list if there are more than one covariate. If
        ``covar`` is not None, a partial correlation will be computed using
        :py:func:`pingouin.partial_corr` function.

        .. important:: Only ``method='pearson'`` and ``method='spearman'``
            are currently supported in partial correlation.
    alternative : string
        Defines the alternative hypothesis, or tail of the correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the correlation is
        positive (greater than zero), "less" tests against the hypothesis that the correlation is
        negative.
    method : string
        Correlation type:

        * ``'pearson'``: Pearson :math:`r` product-moment correlation
        * ``'spearman'``: Spearman :math:`\\rho` rank-order correlation
        * ``'kendall'``: Kendall's :math:`\\tau_B` correlation
          (for ordinal data)
        * ``'bicor'``: Biweight midcorrelation (robust)
        * ``'percbend'``: Percentage bend correlation (robust)
        * ``'shepherd'``: Shepherd's pi correlation (robust)
        * ``'skipped'``: Skipped correlation (robust)
    padjust : string
        Method used for testing and adjustment of pvalues.

        * ``'none'``: no correction
        * ``'bonf'``: one-step Bonferroni correction
        * ``'sidak'``: one-step Sidak correction
        * ``'holm'``: step-down method using Bonferroni adjustments
        * ``'fdr_bh'``: Benjamini/Hochberg FDR correction
        * ``'fdr_by'``: Benjamini/Yekutieli FDR correction
    nan_policy : string
        Can be ``'listwise'`` for listwise deletion of missing values
        (= complete-case analysis) or ``'pairwise'`` (default) for the more
        liberal pairwise deletion (= available-case analysis).

        .. versionadded:: 0.2.9

    Returns
    -------
    stats : :py:class:`pandas.DataFrame`

        * ``'X'``: Name(s) of first columns.
        * ``'Y'``: Name(s) of second columns.
        * ``'method'``: Correlation type.
        * ``'covar'``: List of specified covariate(s), only when covariates are passed.
        * ``'alternative'``: Tail of the test.
        * ``'n'``: Sample size (after removal of missing values).
        * ``'r'``: Correlation coefficients.
        * ``'CI95'``: 95% parametric confidence intervals.
        * ``'p_unc'``: Uncorrected p-values.
        * ``'p_corr'``: Corrected p-values.
        * ``'p_adjust'``: P-values correction method.
        * ``'BF10'``: Bayes Factor of the alternative hypothesis (only for Pearson correlation)
        * ``'power'``: achieved power of the test (= 1 - type II error).

    Notes
    -----
    Please refer to the :py:func:`pingouin.corr()` function for a description
    of the different methods. Missing values are automatically removed from the
    data using a pairwise deletion.

    This function is more flexible and gives a much more detailed
    output than the :py:func:`pandas.DataFrame.corr()` method (i.e. p-values,
    confidence interval, Bayes Factor...). This comes however at
    an increased computational cost. While this should not be discernible for
    a dataframe with less than 10,000 rows and/or less than 20 columns, this
    function can be slow for very large datasets.

    A faster alternative to get the r-values and p-values in a matrix format is
    to use the :py:func:`pingouin.rcorr` function, which works directly as a
    :py:class:`pandas.DataFrame` method (see example below).

    This function also works with two-dimensional multi-index columns. In this
    case, columns must be list(s) of tuple(s). Please refer to this `example
    Jupyter notebook
    <https://github.com/raphaelvallat/pingouin/blob/master/notebooks/04_Correlations.ipynb>`_
    for more details.

    If and only if ``covar`` is specified, this function will compute the
    pairwise partial correlation between the variables. If you are only
    interested in computing the partial correlation matrix (i.e. the raw
    pairwise partial correlation coefficient matrix, without the p-values,
    sample sizes, etc), a better alternative is to use the
    :py:func:`pingouin.pcorr` function (see example 7).

    Examples
    --------
    1. One-sided spearman correlation corrected for multiple comparisons

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> pd.set_option("display.max_columns", 20)
    >>> data = pg.read_dataset("pairwise_corr").iloc[:, 1:]
    >>> pg.pairwise_corr(data, method="spearman", alternative="greater", padjust="bonf").round(3)
                   X                  Y    method alternative    n      r          CI95  p_unc  p_corr p_adjust  power
    0    Neuroticism       Extraversion  spearman     greater  500 -0.325  [-0.39, 1.0]  1.000   1.000     bonf  0.000
    1    Neuroticism           Openness  spearman     greater  500 -0.028   [-0.1, 1.0]  0.735   1.000     bonf  0.012
    2    Neuroticism      Agreeableness  spearman     greater  500 -0.151  [-0.22, 1.0]  1.000   1.000     bonf  0.000
    3    Neuroticism  Conscientiousness  spearman     greater  500 -0.356  [-0.42, 1.0]  1.000   1.000     bonf  0.000
    4   Extraversion           Openness  spearman     greater  500  0.243   [0.17, 1.0]  0.000   0.000     bonf  1.000
    5   Extraversion      Agreeableness  spearman     greater  500  0.062  [-0.01, 1.0]  0.083   0.832     bonf  0.398
    6   Extraversion  Conscientiousness  spearman     greater  500  0.056  [-0.02, 1.0]  0.106   1.000     bonf  0.345
    7       Openness      Agreeableness  spearman     greater  500  0.170    [0.1, 1.0]  0.000   0.001     bonf  0.985
    8       Openness  Conscientiousness  spearman     greater  500 -0.007  [-0.08, 1.0]  0.560   1.000     bonf  0.036
    9  Agreeableness  Conscientiousness  spearman     greater  500  0.161   [0.09, 1.0]  0.000   0.002     bonf  0.976

    2. Robust two-sided biweight midcorrelation with uncorrected p-values

    >>> pcor = pg.pairwise_corr(
    ...     data, columns=["Openness", "Extraversion", "Neuroticism"], method="bicor"
    ... )
    >>> pcor.round(3)
                  X             Y method alternative    n      r            CI95  p_unc  power
    0      Openness  Extraversion  bicor   two-sided  500  0.247    [0.16, 0.33]  0.000  1.000
    1      Openness   Neuroticism  bicor   two-sided  500 -0.028   [-0.12, 0.06]  0.535  0.095
    2  Extraversion   Neuroticism  bicor   two-sided  500 -0.343  [-0.42, -0.26]  0.000  1.000

    3. One-versus-all pairwise correlations

    >>> pg.pairwise_corr(data, columns=["Neuroticism"]).round(3)
                 X                  Y   method alternative    n      r            CI95  p_unc       BF10  power
    0  Neuroticism       Extraversion  pearson   two-sided  500 -0.350  [-0.42, -0.27]  0.000  6.765e+12  1.000
    1  Neuroticism           Openness  pearson   two-sided  500 -0.010    [-0.1, 0.08]  0.817      0.058  0.056
    2  Neuroticism      Agreeableness  pearson   two-sided  500 -0.134  [-0.22, -0.05]  0.003      5.122  0.854
    3  Neuroticism  Conscientiousness  pearson   two-sided  500 -0.368  [-0.44, -0.29]  0.000  2.644e+14  1.000

    4. Pairwise correlations between two lists of columns (cartesian product)

    >>> columns = [["Neuroticism", "Extraversion"], ["Openness"]]
    >>> pg.pairwise_corr(data, columns).round(3)
                  X         Y   method alternative    n      r          CI95  p_unc       BF10  power
    0   Neuroticism  Openness  pearson   two-sided  500 -0.010  [-0.1, 0.08]  0.817      0.058  0.056
    1  Extraversion  Openness  pearson   two-sided  500  0.267  [0.18, 0.35]  0.000  5.277e+06  1.000

    5. As a Pandas method

    >>> pcor = data.pairwise_corr(covar="Neuroticism", method="spearman")

    6. Pairwise partial correlation

    >>> pg.pairwise_corr(data, covar=["Neuroticism", "Openness"])
                   X                  Y   method                        covar alternative    n         r           CI95     p_unc
    0   Extraversion      Agreeableness  pearson  ['Neuroticism', 'Openness']   two-sided  500 -0.038737  [-0.13, 0.05]  0.388361
    1   Extraversion  Conscientiousness  pearson  ['Neuroticism', 'Openness']   two-sided  500 -0.071427  [-0.16, 0.02]  0.111389
    2  Agreeableness  Conscientiousness  pearson  ['Neuroticism', 'Openness']   two-sided  500  0.123108   [0.04, 0.21]  0.005944

    7. Pairwise partial correlation matrix using :py:func:`pingouin.pcorr`

    >>> data[["Neuroticism", "Openness", "Extraversion"]].pcorr().round(3)
                  Neuroticism  Openness  Extraversion
    Neuroticism         1.000     0.092        -0.360
    Openness            0.092     1.000         0.281
    Extraversion       -0.360     0.281         1.000

    8. Correlation matrix with p-values using :py:func:`pingouin.rcorr`

    >>> data[["Neuroticism", "Openness", "Extraversion"]].rcorr()
                 Neuroticism Openness Extraversion
    Neuroticism            -                   ***
    Openness           -0.01        -          ***
    Extraversion       -0.35    0.267            -
    """
    from pingouin.correlation import corr, partial_corr

    # Check arguments
    _check_alternative(alternative)
    assert nan_policy in ["listwise", "pairwise"]

    # Keep only numeric columns
    data = data._get_numeric_data()
    # Remove columns with constant value and/or NaN
    data = data.loc[:, data.nunique(dropna=True) >= 2]
    # Extract columns names
    keys = data.columns.tolist()

    # First ensure that columns is a list
    if isinstance(columns, (str, tuple)):
        columns = [columns]

    # Check if columns index has multiple levels
    if isinstance(data.columns, pd.MultiIndex) and columns is not None:
        # Simple List with one element: [('L0', 'L1')]
        # Simple list with >= 2 elements: [('L0', 'L1'), ('L0', 'L2')]
        # Nested lists: [[('L0', 'L1')], ...] or [..., [('L0', 'L1')]]
        assert all(isinstance(c, tuple) for c in _flatten_list(list(columns)))

    # Then define combinations / products between columns. Pairs are kept as tuples of column
    # labels (not converted to a numpy array), so that column labels of any type are preserved.
    if columns is None:
        # Case A: column is not defined --> corr between all numeric columns
        combs = list(combinations(keys, 2))
    elif isinstance(columns[0], (list, np.ndarray)):
        # Case B: two lists of columns
        group1 = [e for e in columns[0] if e in keys]
        group2 = columns[1] if len(columns) > 1 else None
        if isinstance(group2, (list, np.ndarray)) and len(group2):
            # B1: [['a', 'b'], ['c', 'd']]
            group2 = [e for e in group2 if e in keys]
        else:
            # B2: [['a', 'b']], [['a', 'b'], None] or [['a', 'b'], 'all']
            group2 = [e for e in keys if e not in group1]
        combs = list(product(group1, group2))
    else:
        # Case C: column is a simple list
        if len(columns) == 1 and columns[0] not in keys:
            raise ValueError(f'"{columns[0]}" is not in data or is not numeric.')
        # Make sure that we keep numeric columns
        columns = [c for c in columns if c in keys]
        if len(columns) == 1:
            # C1: one-versus-all, e.g. ['a'] or 'a'
            combs = [(columns[0], e) for e in keys if e != columns[0]]
        else:
            # C2: combinations between ['a', 'b', 'c']
            combs = list(combinations(columns, 2))

    # Now we check if covariates are present
    if covar is not None:
        assert isinstance(covar, (str, list, pd.Index)), "covar must be list or string."
        if isinstance(covar, str):
            covar = [covar]
        elif isinstance(covar, pd.Index):
            covar = covar.tolist()
        # Check that columns exist and are numeric
        assert all([c in keys for c in covar]), (
            "Covariate(s) are either not in data or not numeric."
        )
        # And we make sure that X or Y does not contain covar
        combs = [(x, y) for x, y in combs if x not in covar and y not in covar]

    if len(combs) == 0:
        raise ValueError(
            "No column combination found. Please make sure that the specified columns (and "
            "covar) exist in the dataframe, are numeric, and contains at least two unique values."
        )

    # Listwise deletion of missing values
    if nan_policy == "listwise":
        all_cols = list(dict.fromkeys(c for pair in combs for c in pair))
        if covar is not None:
            all_cols.extend(covar)
        data = data[all_cols].dropna()

    # Compute pairwise correlations
    results = []
    with _no_rounding():  # For max precision
        for col1, col2 in combs:
            if covar is None:
                cor_st = corr(
                    data[col1].to_numpy(),
                    data[col2].to_numpy(),
                    alternative=alternative,
                    method=method,
                )
            else:
                cor_st = partial_corr(
                    data=data, x=col1, y=col2, covar=covar, alternative=alternative, method=method
                )
            results.append(cor_st)

    stats = pd.concat(results, ignore_index=True)
    stats.insert(0, "X", [x for x, _ in combs])
    stats.insert(1, "Y", [y for _, y in combs])
    stats.insert(2, "method", method)
    stats.insert(3, "alternative", alternative)

    # Force conversion to numeric
    dtypes = {"r": float, "n": int, "p_val": float, "outliers": float, "power": float}
    stats = stats.astype({k: v for k, v in dtypes.items() if k in stats.columns})

    # Multiple comparisons
    stats = stats.rename(columns={"p_val": "p_unc"})
    stats = _padjust(stats, padjust=padjust)

    col_order = [
        "X",
        "Y",
        "method",
        "alternative",
        "n",
        "outliers",
        "r",
        "CI95",
        "p_unc",
        "p_corr",
        "p_adjust",
        "BF10",
        "power",
    ]

    # Reorder columns and remove empty ones
    stats = stats.reindex(columns=col_order).dropna(how="all", axis=1)

    # Add covariates names if present
    if covar is not None:
        stats.insert(loc=3, column="covar", value=str(covar))

    return _postprocess_dataframe(stats)
