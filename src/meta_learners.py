"""增益模型（手写）：响应模型、S-learner、T-learner、X-learner、类别转换法。

统一接口：model.fit(X, y, t) 之后，model.predict_uplift(X) 返回每个人的预测增益
（发短信时的购买概率 − 不发短信时的购买概率）。
"""
import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED


# ---------- 基础模型 ----------

def logistic():
    """逻辑回归：先用中位数填补缺失，再标准化（特征量纲差别很大）。"""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))


def lgbm_classifier(**kw):
    """LightGBM 分类器，能直接处理缺失值。"""
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def lgbm_regressor(**kw):
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
    params.update(kw)
    return lgb.LGBMRegressor(**params)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


# ---------- 对照：普通响应模型 ----------

class ResponseModel:
    """普通响应模型：只预测“会不会买”，完全不区分是否收到短信。

    用它的预测值排序，相当于“把短信发给最可能购买的人”，是增益模型要对比的传统做法。
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y, t):
        self.model_ = clone(self.base).fit(X, y)
        return self

    def predict_uplift(self, X):
        return _proba(self.model_, X)  # 返回的是购买概率，不是增益，只用于排序比较


# ---------- 增益模型 ----------

class SLearner:
    """S-learner（单模型）：把“是否收到短信”当作一个普通特征，训练一个模型。

    预测时把同一个人的这个特征分别设为 1 和 0，两次预测之差就是增益。
    问题：如果模型认为这个特征不重要（比如树模型很少用它分裂），预测的增益会被压向 0。
    """

    def __init__(self, base):
        self.base = base

    def _with_t(self, X, value):
        return np.column_stack([np.asarray(X, dtype=float), np.full(len(X), value, dtype=float)])

    def fit(self, X, y, t):
        Xt = np.column_stack([np.asarray(X, dtype=float), np.asarray(t, dtype=float)])
        self.model_ = clone(self.base).fit(Xt, y)
        return self

    def predict_uplift(self, X):
        return _proba(self.model_, self._with_t(X, 1)) - _proba(self.model_, self._with_t(X, 0))


class TLearner:
    """T-learner（双模型）：实验组、对照组各训练一个模型，预测增益 = 实验组模型的概率 − 对照组模型的概率。

    问题：两个模型各自的误差会叠加到增益上；增益通常比购买概率小得多，容易被噪声淹没。
    """

    def __init__(self, base_t, base_c=None):
        self.base_t, self.base_c = base_t, base_c if base_c is not None else base_t

    def fit(self, X, y, t):
        X, y, t = np.asarray(X, dtype=float), np.asarray(y), np.asarray(t)
        self.model_t_ = clone(self.base_t).fit(X[t == 1], y[t == 1])
        self.model_c_ = clone(self.base_c).fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return _proba(self.model_t_, X) - _proba(self.model_c_, X)

    def predict_control(self, X):
        """不发短信时的购买概率（用来识别“铁定买”的人）。"""
        return _proba(self.model_c_, X)


class XLearner:
    """X-learner（Künzel 等，2019）：在 T-learner 的基础上，直接对“个人增益的估计值”建模。

    1. 和 T-learner 一样，分别训练 μ1（实验组）、μ0（对照组）；
    2. 构造每个人的“估算增益”：实验组的人 D1 = 实际结果 − μ0(x)；对照组的人 D0 = μ1(x) − 实际结果；
    3. 分别用回归模型拟合 τ1(x) ≈ D1、τ0(x) ≈ D0；
    4. 合并：τ(x) = g × τ0(x) + (1 − g) × τ1(x)，g 是进入实验组的概率（随机实验里是常数，用训练集的实验组占比）。
    """

    def __init__(self, base_clf, base_reg):
        self.base_clf, self.base_reg = base_clf, base_reg

    def fit(self, X, y, t):
        X, y, t = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(t)
        self.t_learner_ = TLearner(self.base_clf).fit(X, y, t)
        mu1 = _proba(self.t_learner_.model_t_, X)
        mu0 = _proba(self.t_learner_.model_c_, X)
        d1 = y[t == 1] - mu0[t == 1]
        d0 = mu1[t == 0] - y[t == 0]
        self.tau1_ = clone(self.base_reg).fit(X[t == 1], d1)
        self.tau0_ = clone(self.base_reg).fit(X[t == 0], d0)
        self.g_ = float(t.mean())
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        return self.g_ * self.tau0_.predict(X) + (1 - self.g_) * self.tau1_.predict(X)


class ClassTransformation:
    """类别转换法：构造新标签 Z = 1（实验组且购买，或对照组且未购买），否则 Z = 0。

    当进入实验组的概率是 0.5 时，可以证明：增益 = 2 × P(Z = 1 | x) − 1。
    只需要训练一个普通分类器，所以简单、稳定。前提是实验组占比接近 0.5（这里是 0.4998）。
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y, t):
        y, t = np.asarray(y), np.asarray(t)
        z = ((y == 1) & (t == 1)) | ((y == 0) & (t == 0))
        self.model_ = clone(self.base).fit(X, z.astype(int))
        return self

    def predict_uplift(self, X):
        return 2 * _proba(self.model_, X) - 1
