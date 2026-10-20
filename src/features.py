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
