from math import sqrt

import numpy as np
from numpy import mean
from numpy import var
from scipy.stats import wilcoxon
from statsmodels.stats.power import FTestAnovaPower


def cohend(d1, d2):
    """
    function to calculate Cohen's d for independent samples
    """

    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    d = (u1 - u2) / s
    d = abs(d)

    result = ''
    if d < 0.2:
        result = 'negligible'
    if 0.2 <= d < 0.5:
        result = 'small'
    if 0.5 <= d < 0.8:
        result = 'medium'
    if d >= 0.8:
        result = 'large'

    return result, d


def run_power_analysis_two_sets(simulation1, simulation2):
    simulation1 = np.round(simulation1, decimals=3)
    simulation2 = np.round(simulation2, decimals=3)

    eff_size = cohend(simulation1, simulation2)

    pow = FTestAnovaPower().solve_power(effect_size=eff_size[1], nobs=len(simulation1) + len(simulation2), alpha=0.05)
    nobs = FTestAnovaPower().solve_power(effect_size=eff_size[1], power=0.8, alpha=0.05)

    print(f"Pow: {pow}")
    print(f"Nobs: {nobs}\n")

    return pow


def run_wilcoxon_and_cohend(data1, data2):
    w_statistic, pvalue = wilcoxon(data1, data2)
    cohensd = cohend(data1, data2)
    print(f"Sum of ranks of differences is: {w_statistic}")
    print(f"P-Value is: {pvalue}")
    print(f"Cohen's D is: {cohensd}")

    return pvalue, cohensd[0]
