# 数据说明

原始数据放在 `data/raw/x5/`，处理后的数据放在 `data/processed/`，两者都不提交到 Git。原始数据仅用于学习，不再分发。

## X5 RetailHero Uplift Modeling
- 比赛页面：https://ods.ai/competitions/x5-retailhero-uplift-modeling（2019 年，X5 Retail Group 举办，已结束）
- 下载镜像：scikit-uplift 维护的公开镜像（无需登录）。数据字段说明：https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html

| 文件 | 镜像地址 | 大小（服务器返回，2026-10-20 查看） |
|---|---|---|
| `uplift_train.csv.gz` | https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz | 1,182,423 字节 |
| `clients.csv.gz` | https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz | 7,637,755 字节 |
| `purchases.csv.gz` | https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz | 669,979,268 字节 |

### 官方 / 文档说明（下载后会用代码逐项核实）
- `uplift_train`：带实验标签的客户，`treatment_flg` 表示是否收到短信，`target` 表示之后是否购买。scikit-uplift 文档写的是 200,039 行，处理比例 0.5，响应比例 0.62。
- `clients`：客户基本信息，文档写的是 400,162 行。
- `purchases`：客户在**短信发送之前**的交易流水，文档写的是 45,786,568 行，解压后合计约 4.17 GB。
- 比赛页面只说 X5 用短信“鼓励客户多购物”，**没有说明分组是否随机**，也没有说明短信内容和购买的统计期。随机化要在阶段 1 自己验证。

### 状态
已下载（2026-10-20，用户同意后从上述镜像下载），保存在 `data/raw/x5/`。校验与读取代码见 `src/data.py`。

| 文件 | 字节数 | SHA256 |
|---|---|---|
| `uplift_train.csv.gz` | 1,182,423 | `0a218e864a1ec1dfe26e980a082b81d5f46d199c4a6ae32abbaa95265af1c2c7` |
| `clients.csv.gz` | 7,637,755 | `bfca1719b862735ae9816ed9bbba6924931c6a4dff761c481efe9addff458429` |
| `purchases.csv.gz` | 669,979,268 | `3d57519fa92b553fd1e947c120add475cc42e0c64c323a52cef5aebe8645c8e2` |

已核实（`results/stage1_ab_test.json`）：实验标签表 200,039 行、客户信息表 400,162 行、交易流水 45,786,568 行，和文档一致。交易流水时间范围为 2018-11-21 21:02:33 至 2019-03-18 23:40:03，本项目据此把截止时间定为 2019-03-19 0 点。

交易流水另存为 `data/processed/purchases.parquet`（由 `src/data.py` 生成，不进 Git）。

### 已知问题
- 客户年龄有异常值（最小 −7491、最大 1901），不在 10 到 100 岁之间的 1,404 人按缺失处理。
- 首次兑换积分日期最晚到 2019-11-20，其中 44,953 人在截止日之后才首次兑换。这部分信息发生在短信之后，构造特征时截断到截止日之前。
- 536 人的首次兑换早于首次办卡，保留原样并记录。
- 28 笔交易的交易金额在不同行之间不一致，取最大值。
