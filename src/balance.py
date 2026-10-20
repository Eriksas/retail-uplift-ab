"""随机分组平衡性检验：标准化均值差（SMD）和联合检验。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def smd(x_t, x_c):
    """标准化均值差：两组均值之差 / 合并标准差。

    合并标准差 = sqrt((处理组样本方差 + 对照组样本方差) / 2)，样本方差用 ddof=1。
    这与 Stata pstest 的 %bias 口径一致（%bias = SMD × 100）。
    """
    x_t = np.asarray(x_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    pooled_sd = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return np.nan  # 两组都是常数时 SMD 没有定义
    return (x_t.mean() - x_c.mean()) / pooled_sd


def expand_features(df, numeric, categorical, drop_first=False):
    """数值和 0/1 变量原样保留，类别变量展开成哑变量（列名形如 channel=Web）。

    drop_first=False：保留每个取值，用于逐个检查平衡。
    drop_first=True：去掉每个类别变量的第一个取值作为基准类，用于回归，避免完全共线。
    """
    dummies = pd.get_dummies(df[categorical], prefix_sep="=", drop_first=drop_first, dtype=float)
    return pd.concat([df[numeric].astype(float), dummies], axis=1)


def balance_table(X, treat):
    """逐列比较处理组（treat=1）和对照组（treat=0）：均值、SMD、Welch t 检验 p 值。"""
    rows = []
    for col in X.columns:
        x_t = X.loc[treat == 1, col]
        x_c = X.loc[treat == 0, col]
        rows.append({
            "feature": col,
            "mean_treat": x_t.mean(),
            "mean_control": x_c.mean(),
            "smd": smd(x_t, x_c),
            "p_value": stats.ttest_ind(x_t, x_c, equal_var=False).pvalue,
        })
    return pd.DataFrame(rows).set_index("feature")


def joint_balance_test(X, treat):
    """联合检验：用全部特征做 logit 回归，预测是否被分到处理组。

    似然比检验（LR test）的原假设是“所有特征的系数同时为 0”，也就是特征完全无法预测分组。
    X 中的类别变量要先去掉基准类（expand_features(..., drop_first=True)）。
    """
    model = sm.Logit(treat.astype(float), sm.add_constant(X)).fit(disp=0)
    return {
        "lr_stat": float(model.llr),
        "df": int(model.df_model),
        "p_value": float(model.llr_pvalue),
        "pseudo_r2": float(model.prsquared),
    }
