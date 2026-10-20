"""原始数据的校验与读取。原始数据只读，不在这里做任何清洗。

交易流水约 4,500 万行，不读进 pandas，而是先用 DuckDB 转存为 Parquet（列式存储，读取快、占内存少），
之后所有聚合都用 SQL 在 DuckDB 里完成。
"""
import hashlib

import duckdb
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW

RAW_DIR = DATA_RAW / "x5"
FILES = {
    "uplift_train": RAW_DIR / "uplift_train.csv.gz",
    "clients": RAW_DIR / "clients.csv.gz",
    "purchases": RAW_DIR / "purchases.csv.gz",
}
SHA256 = {  # 2026-10-20 首次下载时计算，见 data/README.md
    "uplift_train": "0a218e864a1ec1dfe26e980a082b81d5f46d199c4a6ae32abbaa95265af1c2c7",
    "clients": "bfca1719b862735ae9816ed9bbba6924931c6a4dff761c481efe9addff458429",
    "purchases": "3d57519fa92b553fd1e947c120add475cc42e0c64c323a52cef5aebe8645c8e2",
}
PURCHASES_PARQUET = DATA_PROCESSED / "purchases.parquet"


def sha256(path):
    """计算文件的 SHA256，用来确认拿到的是同一份数据。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files():
    """校验三个原始文件的 SHA256（已记录的才校验），返回 {文件名: (字节数, sha256)}。"""
    out = {}
    for name, path in FILES.items():
        digest = sha256(path)
        if name in SHA256 and SHA256[name] != digest:
            raise ValueError(f"校验失败：{path.name} 的 SHA256 为 {digest}")
        out[name] = (path.stat().st_size, digest)
    return out


def load_train():
    """带实验标签的客户：client_id、treatment_flg（是否收到短信）、target（之后是否购买）。"""
    return pd.read_csv(FILES["uplift_train"])


def load_clients():
    """客户基本信息。日期列转成时间类型。"""
    df = pd.read_csv(FILES["clients"])
    for col in ["first_issue_date", "first_redeem_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def purchases_parquet():
    """把交易流水转存为 Parquet（已存在则跳过），返回路径。"""
    if not PURCHASES_PARQUET.exists():
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = PURCHASES_PARQUET.with_suffix(".part")
        con = duckdb.connect()
        con.execute("SET preserve_insertion_order = false")  # 降低内存占用
        con.execute(f"""
            COPY (SELECT * FROM read_csv_auto('{FILES["purchases"].as_posix()}', header = true))
            TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)
        con.close()
        tmp.rename(PURCHASES_PARQUET)
    return PURCHASES_PARQUET


def connect():
    """返回一个 DuckDB 连接，并注册视图 purchases、train、clients，方便直接写 SQL。"""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW purchases AS SELECT * FROM read_parquet('{purchases_parquet().as_posix()}')")
    con.execute(f"CREATE VIEW train AS SELECT * FROM read_csv_auto('{FILES['uplift_train'].as_posix()}')")
    con.execute(f"CREATE VIEW clients AS SELECT * FROM read_csv_auto('{FILES['clients'].as_posix()}')")
    return con
