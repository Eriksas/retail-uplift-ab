"""从交易流水和客户信息中构造客户特征（全部用 DuckDB SQL 完成）。

交易流水每一行是一笔交易里的一件商品；purchase_sum、积分等字段是整笔交易的值，在同一笔交易的每一行都重复出现。
所以“交易层面”的统计要先按 transaction_id 去重，再按客户汇总，否则金额会被重复计算。

截止时间 CUTOFF：流水的最后一条记录是 2019-03-18 23:40:03，短信在此之后发送。
所有特征都只能使用 CUTOFF 之前的信息，否则会用到受短信影响的数据（信息泄露）。
"""
import pandas as pd

CUTOFF = "2019-03-19 00:00:00"

# 交易层面的中间表：一笔交易一行
TRANSACTIONS_SQL = f"""
    SELECT client_id, transaction_id,
           MIN(transaction_datetime)     AS dt,
           MAX(purchase_sum)             AS purchase_sum,
           MAX(regular_points_received)  AS regular_points_received,
           MAX(express_points_received)  AS express_points_received,
           MAX(regular_points_spent)     AS regular_points_spent,
           MAX(express_points_spent)     AS express_points_spent,
           ANY_VALUE(store_id)           AS store_id,
           COUNT(*)                      AS n_lines,
           SUM(product_quantity)         AS n_items
    FROM purchases
    WHERE transaction_datetime < TIMESTAMP '{CUTOFF}'
    GROUP BY client_id, transaction_id
"""


def pre_period_metrics(con):
    """阶段 1 用的几个实验前指标：交易次数、消费总额、客单价、最近一次购买距截止日的天数、最近 7/14/30 天的交易次数。"""
    sql = f"""
        WITH trn AS ({TRANSACTIONS_SQL})
        SELECT client_id,
               COUNT(*)                                                          AS n_trans,
               SUM(purchase_sum)                                                 AS total_sum,
               AVG(purchase_sum)                                                 AS avg_sum,
               date_diff('day', MAX(dt), TIMESTAMP '{CUTOFF}')                   AS recency_days,
               COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 7 DAY)  AS n_trans_7d,
               COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 14 DAY) AS n_trans_14d,
               COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY) AS n_trans_30d
        FROM trn
        GROUP BY client_id
    """
    return con.sql(sql).df().set_index("client_id")


def client_basic(clients):
    """客户信息表的清洗版特征（只用截止时间之前的信息）。

    - 年龄：不在 [10, 100] 内的视为异常，设为缺失
    - 首次办卡、首次兑换积分距截止日的天数；首次兑换发生在截止日之后的，视为“截止日前未兑换”
    """
    cutoff = pd.Timestamp(CUTOFF)
    out = pd.DataFrame(index=clients["client_id"])
    age = clients["age"].to_numpy()
    out["age"] = pd.Series(age, index=out.index).where((age >= 10) & (age <= 100))
    out["age_missing"] = out["age"].isna().astype(int)
    for g in ["F", "M", "U"]:
        out[f"gender_{g}"] = (clients["gender"].to_numpy() == g).astype(int)
    issue = clients["first_issue_date"].to_numpy()
    redeem = clients["first_redeem_date"]
    redeem_before = redeem.where(redeem < cutoff)
    out["days_since_issue"] = (cutoff - pd.Series(issue, index=out.index)).dt.days
    out["redeemed_before_cutoff"] = redeem_before.notna().to_numpy().astype(int)
    out["days_since_first_redeem"] = (cutoff - pd.Series(redeem_before.to_numpy(), index=out.index)).dt.days
    out["issue_to_redeem_days"] = (pd.Series(redeem_before.to_numpy(), index=out.index)
                                   - pd.Series(issue, index=out.index)).dt.days
    return out


# ---------- 阶段 2：完整的客户特征 ----------

FEATURES_SQL = f"""
WITH trn AS ({TRANSACTIONS_SQL}),
lines AS (
    SELECT client_id,
           COUNT(DISTINCT product_id)                                   AS n_products,
           SUM(trn_sum_from_iss)                                        AS line_sum,
           SUM(COALESCE(trn_sum_from_red, 0))                           AS line_sum_paid_by_points,
           AVG(CASE WHEN trn_sum_from_red IS NOT NULL THEN 1 ELSE 0 END) AS share_lines_paid_by_points
    FROM purchases
    WHERE transaction_datetime < TIMESTAMP '{CUTOFF}'
    GROUP BY client_id
),
store_counts AS (
    SELECT client_id, store_id, COUNT(*) AS k FROM trn GROUP BY client_id, store_id
),
store AS (
    SELECT client_id, COUNT(*) AS n_stores, MAX(k) AS top_store_trans FROM store_counts GROUP BY client_id
),
agg AS (
    SELECT client_id,
           -- 频率与金额
           COUNT(*)                                       AS n_trans,
           SUM(purchase_sum)                              AS total_sum,
           AVG(purchase_sum)                              AS avg_sum,
           STDDEV_SAMP(purchase_sum)                      AS std_sum,
           MAX(purchase_sum)                              AS max_sum,
           MEDIAN(purchase_sum)                           AS median_sum,
           COUNT(DISTINCT CAST(dt AS DATE))               AS n_active_days,
           -- 时间跨度与近期活跃度
           date_diff('day', MAX(dt), TIMESTAMP '{CUTOFF}')  AS recency_days,
           date_diff('day', MIN(dt), TIMESTAMP '{CUTOFF}')  AS first_purchase_days,
           COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 7 DAY)   AS n_trans_7d,
           COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 14 DAY)  AS n_trans_14d,
           COUNT(*) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY)  AS n_trans_30d,
           COALESCE(SUM(purchase_sum) FILTER (WHERE dt >= TIMESTAMP '{CUTOFF}' - INTERVAL 30 DAY), 0) AS sum_30d,
           -- 积分与促销
           SUM(regular_points_received)                   AS regular_points_received,
           SUM(express_points_received)                   AS express_points_received,
           SUM(regular_points_spent)                      AS regular_points_spent,
           SUM(express_points_spent)                      AS express_points_spent,
           AVG(CASE WHEN express_points_received > 0 THEN 1 ELSE 0 END) AS share_trans_express_received,
           AVG(CASE WHEN regular_points_spent < 0 OR express_points_spent < 0 THEN 1 ELSE 0 END) AS share_trans_points_spent,
           -- 购物篮
           AVG(n_items)                                   AS avg_items,
           AVG(n_lines)                                   AS avg_lines,
           -- 购物时间习惯
           AVG(CASE WHEN isodow(dt) >= 6 THEN 1 ELSE 0 END) AS share_weekend,
           AVG(CASE WHEN hour(dt) >= 18 THEN 1 ELSE 0 END)  AS share_evening,
           AVG(CASE WHEN hour(dt) < 12 THEN 1 ELSE 0 END)   AS share_morning,
           AVG(hour(dt))                                  AS mean_hour
    FROM trn
    GROUP BY client_id
)
SELECT agg.*, lines.n_products, lines.share_lines_paid_by_points,
       lines.line_sum_paid_by_points / NULLIF(lines.line_sum, 0) AS share_sum_paid_by_points,
       store.n_stores, store.top_store_trans * 1.0 / agg.n_trans    AS top_store_share
FROM agg
JOIN lines USING (client_id)
JOIN store USING (client_id)
"""


def build_features(con, clients):
    """所有客户的特征表：交易流水特征（SQL）+ 客户信息特征，再加几个比例特征。"""
    f = con.sql(FEATURES_SQL).df().set_index("client_id")
    f["share_trans_30d"] = f["n_trans_30d"] / f["n_trans"]
    f["share_sum_30d"] = f["sum_30d"] / f["total_sum"]
    span = (f["first_purchase_days"] - f["recency_days"]).clip(lower=1)
    f["avg_interval_days"] = span / (f["n_trans"] - 1).clip(lower=1)
    f["trans_per_week"] = f["n_trans"] / (f["first_purchase_days"] / 7)
    # 积分支付金额字段的口径不清楚，少数客户的占比超过 1；按含义截断到 [0, 1]
    f["share_sum_paid_by_points"] = f["share_sum_paid_by_points"].clip(0, 1)
    return f.join(client_basic(clients), how="left")


# 特征说明：(分组, 含义)
FEATURE_DESC = {
    "n_trans": ("频率与金额", "截止日前的交易次数"),
    "total_sum": ("频率与金额", "消费总额"),
    "avg_sum": ("频率与金额", "平均每笔交易金额（客单价）"),
    "std_sum": ("频率与金额", "交易金额的标准差（只有 1 笔交易时为空）"),
    "max_sum": ("频率与金额", "单笔最大交易金额"),
    "median_sum": ("频率与金额", "交易金额中位数"),
    "n_active_days": ("频率与金额", "有交易的天数"),
    "trans_per_week": ("频率与金额", "平均每周交易次数（从第一次交易算起）"),
    "avg_interval_days": ("频率与金额", "平均交易间隔天数"),
    "recency_days": ("近期活跃度", "最近一次交易距截止日的天数"),
    "first_purchase_days": ("近期活跃度", "流水中第一次交易距截止日的天数"),
    "n_trans_7d": ("近期活跃度", "最近 7 天交易次数"),
    "n_trans_14d": ("近期活跃度", "最近 14 天交易次数"),
    "n_trans_30d": ("近期活跃度", "最近 30 天交易次数"),
    "sum_30d": ("近期活跃度", "最近 30 天消费额"),
    "share_trans_30d": ("近期活跃度", "最近 30 天交易次数占全部交易的比例（趋势）"),
    "share_sum_30d": ("近期活跃度", "最近 30 天消费额占消费总额的比例（趋势）"),
    "regular_points_received": ("积分与促销", "获得的普通积分合计"),
    "express_points_received": ("积分与促销", "获得的活动积分合计"),
    "regular_points_spent": ("积分与促销", "花掉的普通积分合计（负数）"),
    "express_points_spent": ("积分与促销", "花掉的活动积分合计（负数）"),
    "share_trans_express_received": ("积分与促销", "获得活动积分的交易占比"),
    "share_trans_points_spent": ("积分与促销", "使用积分的交易占比（促销敏感度）"),
    "share_lines_paid_by_points": ("积分与促销", "用积分支付的商品行占比"),
    "share_sum_paid_by_points": ("积分与促销", "用积分支付的金额占比（截断到 0–1）"),
    "avg_items": ("购物篮", "平均每笔交易的商品件数"),
    "avg_lines": ("购物篮", "平均每笔交易的商品种类数"),
    "n_products": ("购物篮", "买过的不同商品数"),
    "n_stores": ("门店", "去过的门店数"),
    "top_store_share": ("门店", "最常去门店的交易占比（门店忠诚度）"),
    "share_weekend": ("购物时间", "周末交易占比"),
    "share_evening": ("购物时间", "18 点以后交易占比"),
    "share_morning": ("购物时间", "12 点以前交易占比"),
    "mean_hour": ("购物时间", "平均交易小时"),
    "age": ("客户信息", "年龄（10–100 岁以外设为缺失）"),
    "age_missing": ("客户信息", "年龄是否缺失"),
    "gender_F": ("客户信息", "性别为女"),
    "gender_M": ("客户信息", "性别为男"),
    "gender_U": ("客户信息", "性别未知"),
    "days_since_issue": ("客户信息", "办卡距截止日的天数"),
    "redeemed_before_cutoff": ("客户信息", "截止日前是否兑换过积分"),
    "days_since_first_redeem": ("客户信息", "首次兑换距截止日的天数（截止日前未兑换为空）"),
    "issue_to_redeem_days": ("客户信息", "办卡到首次兑换的天数（截止日前未兑换为空）"),
}
