"""A/B 实验分析（手写）：样本比例失衡检验、平均处理效应、CUPED、最小可检测效应。"""
import numpy as np
import statsmodels.api as sm
from scipy import stats


def srm_test(n_treat, n_control, expected_share=0.5):
    """样本比例失衡检验（SRM）：卡方拟合优度检验，原假设是“分组人数符合设计比例”。"""
    n = n_treat + n_control
    expected = [n * expected_share, n * (1 - expected_share)]
    chi2, p = stats.chisquare([n_treat, n_control], f_exp=expected)
    return {"实验组人数": int(n_treat), "对照组人数": int(n_control), "实验组占比": n_treat / n,
            "设计比例（假设）": expected_share, "卡方统计量": float(chi2), "p值": float(p)}


def diff_in_means(y, t, alpha=0.05):
    """平均处理效应（ATE）= 实验组均值 − 对照组均值。

    标准误用两组各自的方差计算（不假设方差相等）：SE = sqrt(s1²/n1 + s0²/n0)。
    样本量很大时用正态分布构造置信区间和 p 值。
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    q = stats.norm.ppf(1 - alpha / 2)
    z = ate / se
    return {"实验组均值": float(y1.mean()), "对照组均值": float(y0.mean()), "ATE": float(ate),
            "标准误": float(se), "置信区间下限": float(ate - q * se), "置信区间上限": float(ate + q * se),
            "z": float(z), "p值": float(2 * stats.norm.sf(abs(z))), "相对提升": float(ate / y0.mean())}


def ols_ate(y, t, covariates=None):
    """回归估计 ATE：y ~ t（+ 协变量），HC1 稳健标准误，相当于 Stata 的 `reg y t x, robust`。"""
    X = np.column_stack([np.asarray(t, dtype=float)] + ([] if covariates is None else [np.asarray(covariates, dtype=float)]))
    res = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X)).fit(cov_type="HC1")
    low, high = res.conf_int()[1]
    return {"ATE": float(res.params[1]), "标准误": float(res.bse[1]), "p值": float(res.pvalues[1]),
            "置信区间下限": float(low), "置信区间上限": float(high)}


def cuped(y, t, x, alpha=0.05):
    """CUPED：用实验前的指标 x 降低结果 y 的方差。

    y_cuped = y − θ × (x − x 的均值)，θ = cov(y, x) / var(x)，用两组合并的数据估计。
    因为 x 发生在实验之前，和分组无关，所以调整后两组均值之差的期望不变（无偏），
    但方差变小了，方差缩减的比例约等于 y 和 x 相关系数的平方。
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    res = diff_in_means(y_adj, t, alpha)
    res["theta"] = float(theta)
    res["方差缩减比例"] = float(1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1))
    res["相关系数平方"] = float(np.corrcoef(y, x)[0, 1] ** 2)
    return res


def mde(se, alpha=0.05, power=0.8):
    """最小可检测效应（MDE）：在给定标准误下，有 power 的概率检测出来的最小真实效应。

    MDE = (z_{1−α/2} + z_{power}) × SE。标准误越小，能检测出的效应越小。
    """
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
