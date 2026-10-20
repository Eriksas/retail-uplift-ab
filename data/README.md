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
待下载（需要用户同意）。
