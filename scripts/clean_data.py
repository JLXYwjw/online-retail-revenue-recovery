# -*- coding: utf-8 -*-
"""清洗 Online_Retail：行级清洗 + 发票层聚合 + 客户 RFM(特征期/标签期时间切分)。"""
import json
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "Online_Retail.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 读原始 ----
df = pd.read_csv(CSV, encoding="latin1")
n0 = len(df)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["revenue"] = df["Quantity"] * df["UnitPrice"]
df["is_cancel"] = df["InvoiceNo"].str.startswith("C")
df["is_return"] = df["revenue"] < 0
df["is_refund_price"] = df["UnitPrice"] < 0
df["has_customer"] = df["CustomerID"].notna()

# ---- 行级清洗表（保留匿名交易，仅过滤日期解析失败） ----
clean = df[["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
            "UnitPrice", "CustomerID", "Country", "revenue",
            "is_cancel", "is_return", "is_refund_price", "has_customer"]]
clean = clean[clean["InvoiceDate"].notna()]
clean["Description"] = clean["Description"].fillna("")
clean.to_parquet(OUT / "retail_clean.parquet", index=False)

# ---- 发票层聚合 ----
inv = clean.groupby("InvoiceNo").agg(
    invoice_date=("InvoiceDate", "first"),
    customer_id=("CustomerID", "first"),
    country=("Country", "first"),
    amount=("revenue", "sum"),
    lines=("StockCode", "size"),
    is_cancel=("is_cancel", "first"),
).reset_index()
inv.to_parquet(OUT / "invoice_level.parquet", index=False)

# ---- 客户 RFM：特征期=2010 全年，标签期=2011 全年（时间切分，无泄漏） ----
feat_end = pd.Timestamp("2011-01-01")
label_end = pd.Timestamp("2012-01-01")
cust = clean[clean["has_customer"]]
feat = cust[cust["InvoiceDate"] < feat_end]
labels = cust[(cust["InvoiceDate"] >= feat_end) & (cust["InvoiceDate"] < label_end)]


def rfm(g):
    return pd.Series({
        "monetary_feat": float(g["revenue"].sum()),
        "freq_feat": int(g["InvoiceNo"].nunique()),
        "recency_feat": float((feat_end - g["InvoiceDate"].max()).days),
        "lines_feat": int(len(g)),
        "return_amt_feat": float(-g.loc[g["is_return"], "revenue"].sum()),
    })


rfm_df = feat.groupby("CustomerID").apply(rfm).reset_index()
lab = labels.groupby("CustomerID").agg(
    future_revenue=("revenue", "sum"),
    future_orders=("InvoiceNo", "nunique"),
).reset_index()
lab["repurchased"] = (lab["future_orders"] > 0).astype(int)
cust_rfm = rfm_df.merge(lab, on="CustomerID", how="left").fillna(
    {"future_revenue": 0.0, "future_orders": 0, "repurchased": 0})
cust_rfm.to_parquet(OUT / "customer_rfm.parquet", index=False)

report = {
    "raw_rows": int(n0),
    "clean_rows": int(len(clean)),
    "cancel_rows": int(clean["is_cancel"].sum()),
    "return_rows": int(clean["is_return"].sum()),
    "no_customer_rows": int((~clean["has_customer"]).sum()),
    "no_customer_pct": round((~clean["has_customer"]).mean() * 100, 1),
    "invoices": int(len(inv)),
    "customers_total": int(clean["CustomerID"].nunique()),
    "customers_feat_period": int(len(cust_rfm)),
    "repurchase_rate_pct": round(float(cust_rfm["repurchased"].mean() * 100), 1),
    "feat_period": "2010-01-01..2010-12-31",
    "label_period": "2011-01-01..2011-12-31",
}
with open(OUT / "cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False))
