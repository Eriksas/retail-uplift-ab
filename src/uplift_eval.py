"""增益模型评估（手写）：增益曲线、Qini 曲线、标准化曲线下面积、前 k% 增益。

为什么需要专门的指标：每个人只能被观察到“发了”或“没发”其中一种结果，个人的真实增益永远看不到，
所以不能像分类那样算准确率。思路是：按模型预测的增益从高到低排序，看排在前面的人群里，
实验组和对照组的购买率差得有多大。如果模型排序有用，前面的人群差距应该明显大于随机排序。

定义和 scikit-uplift 0.5.1 保持一致，便于逐点对比验证。
"""
import numpy as np


def _cumulative(y, uplift, t):
    """按预测增益从高到低排序，在“预测值发生变化”的位置取累计量（并列的预测值作为一个整体处理）。"""
    y, uplift, t = (np.asarray(a, dtype=float) for a in (y, uplift, t))
    order = np.argsort(uplift, kind="mergesort")[::-1]
    y, uplift, t = y[order], uplift[order], t[order]
    cut = np.r_[np.where(np.diff(uplift))[0], len(uplift) - 1]
    n_all = cut + 1
    y_t = np.cumsum(y * t)[cut]
    y_c = np.cumsum(y * (1 - t))[cut]
    n_t = np.cumsum(t)[cut]
    n_c = n_all - n_t
    return n_all, y_t, y_c, n_t, n_c


def _safe_div(a, b):
    """a / b，分母为 0 的位置记为 0。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def uplift_curve(y, uplift, t):
    """增益曲线：横轴是“发给前 n 个人”，纵轴是 (前 n 人中实验组购买率 − 对照组购买率) × n。

    纵轴可以理解为：如果把短信发给这 n 个人，预计能多带来多少次购买。
    """
    n_all, y_t, y_c, n_t, n_c = _cumulative(y, uplift, t)
    # 某一段里只有实验组（或只有对照组）时，缺的那一组购买率按 0 处理（与 scikit-uplift 一致）
    values = (_safe_div(y_t, n_t) - _safe_div(y_c, n_c)) * n_all
    return np.r_[0, n_all], np.r_[0, values]


def qini_curve(y, uplift, t):
    """Qini 曲线：纵轴是“前 n 人中实验组的购买数 − 对照组购买数按人数比例缩放后的值”。

    Qini = Y_T(n) − Y_C(n) × N_T(n) / N_C(n)，和增益曲线思路一样，只是换了一种缩放方式。
    """
    n_all, y_t, y_c, n_t, n_c = _cumulative(y, uplift, t)
    values = y_t - y_c * _safe_div(n_t, n_c)
    return np.r_[0, n_all], np.r_[0, values]


def _perfect_uplift_scores(y, t):
    """“完美模型”的排序分数（和 scikit-uplift 的 perfect_uplift_curve 相同）。"""
    y, t = np.asarray(y), np.asarray(t)
    cr_num = np.sum((y == 1) & (t == 0))
    tn_num = np.sum((y == 0) & (t == 1))
    summand = y if cr_num > tn_num else t
    return 2 * (y == t) + summand


def _area(x, yv):
    return float(np.trapezoid(yv, x))


def uplift_auc(y, uplift, t):
    """标准化的增益曲线下面积：(模型 − 随机) / (完美 − 随机)。0 表示和随机一样，1 表示完美。"""
    x, v = uplift_curve(y, uplift, t)
    xp, vp = uplift_curve(y, _perfect_uplift_scores(y, t), t)
    random_area = _area([0, xp[-1]], [0, vp[-1]])
    return (_area(x, v) - random_area) / (_area(xp, vp) - random_area)


def qini_auc(y, uplift, t):
    """标准化的 Qini 曲线下面积（Qini 系数），完美模型允许存在负增益的人群。"""
    y, t = np.asarray(y), np.asarray(t)
    x, v = qini_curve(y, uplift, t)
    xp, vp = qini_curve(y, y * t - y * (1 - t), t)
    random_area = _area([0, xp[-1]], [0, vp[-1]])
    return (_area(x, v) - random_area) / (_area(xp, vp) - random_area)


def uplift_at_k(y, uplift, t, k=0.3, strategy="overall"):
    """前 k% 增益：只看预测增益最高的前 k% 的人，实验组购买率 − 对照组购买率。

    strategy="overall"：在全体样本里取前 k%；strategy="by_group"：在实验组、对照组里各自取前 k%。
    """
    y, uplift, t = np.asarray(y, dtype=float), np.asarray(uplift, dtype=float), np.asarray(t)
    if strategy == "overall":
        n_top = int(len(y) * k)
        top = np.argsort(uplift, kind="mergesort")[::-1][:n_top]
        yt, tt = y[top], t[top]
        return float(yt[tt == 1].mean() - yt[tt == 0].mean())
    out = []
    for g in (1, 0):
        idx = np.where(t == g)[0]
        n_top = int(len(idx) * k)
        top = idx[np.argsort(uplift[idx], kind="mergesort")[::-1][:n_top]]
        out.append(y[top].mean())
    return float(out[0] - out[1])


def uplift_by_bin(y, uplift, t, n_bins=10):
    """按预测增益分成 n_bins 组（第 1 组预测增益最高），每组计算实际的“实验组 − 对照组”购买率差。

    这是检验模型排序是否靠谱的最直观方法：好的模型，第 1 组的实际增益应该最高，并依次下降。
    """
    y, uplift, t = np.asarray(y, dtype=float), np.asarray(uplift, dtype=float), np.asarray(t)
    order = np.argsort(uplift, kind="mergesort")[::-1]
    rows = []
    for i, idx in enumerate(np.array_split(order, n_bins), start=1):
        yt, tt = y[idx], t[idx]
        p1, p0 = yt[tt == 1].mean(), yt[tt == 0].mean()
        se = np.sqrt(p1 * (1 - p1) / (tt == 1).sum() + p0 * (1 - p0) / (tt == 0).sum())
        rows.append({"组别": i, "人数": len(idx), "预测增益均值": float(uplift[idx].mean()),
                     "实验组购买率": float(p1), "对照组购买率": float(p0), "实际增益": float(p1 - p0), "实际增益标准误": float(se)})
    return rows
