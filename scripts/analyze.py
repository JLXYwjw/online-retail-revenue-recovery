# -*- coding: utf-8 -*-
"""电商指标：收入对账、月度、客户集中度、RFM分层。"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "output"

clean = pd.read_parquet(OUT / "retail_clean.parquet")
inv = pd.read_parquet(OUT / "invoice_level.parquet")
cust = pd.read_parquet(OUT / "customer_rfm.parquet")

m = {}

# ---- 收入对账 ----
pos = clean[clean["revenue"] > 0]
neg = clean[clean["revenue"] < 0]
m["gross_revenue"] = round(float(pos["revenue"].sum()), 2)          # 应确认营收
m["return_leakage"] = round(float(neg["revenue"].sum()), 2)         # 退货/取消损失(负)
m["net_revenue"] = round(float(clean["revenue"].sum()), 2)
m["return_pct"] = round(float(-neg["revenue"].sum() / pos["revenue"].sum() * 100), 1)
m["cancel_invoices"] = int(inv["is_cancel"].sum())
m["cancel_amount"] = round(float(inv.loc[inv["is_cancel"], "amount"].sum()), 2)

# ---- 月度营收与退货 ----
clean = clean.assign(ym=clean["InvoiceDate"].dt.to_period("M"))
mo = clean.groupby("ym").agg(revenue=("revenue", "sum")).reset_index()
mo["ym"] = mo["ym"].astype(str)
m["monthly_revenue"] = mo.to_dict("records")

# 退货按月
ret_mo = neg.assign(ym=neg["InvoiceDate"].dt.to_period("M")).groupby("ym").apply(
    lambda g: float(-g["revenue"].sum()), include_groups=False).reset_index()
ret_mo.columns = ["ym", "return_amt"]
ret_mo["ym"] = ret_mo["ym"].astype(str)
m["monthly_return"] = ret_mo.to_dict("records")

# ---- 国家 ----
m["country_top"] = clean.groupby("Country")["revenue"].sum().sort_values(ascending=False).head(8).astype(float).round(2).to_dict()

# ---- 客户集中度（全部有ID客户按总消费） ----
cc = clean[clean["has_customer"]].groupby("CustomerID")["revenue"].sum().sort_values(ascending=False)
tot = cc.sum()
for p in (5, 10, 20):
    k = max(1, int(len(cc) * p / 100))
    m[f"top{p}pct_share"] = round(float(cc.head(k).sum() / tot * 100), 1)
m["customers_with_id"] = int(len(cc))

# ---- 客户 RFM 分层（特征期 948 客户） ----
def tier(df, col, labels, quantiles):
    try:
        qs = df[col].quantile(quantiles).tolist()
        return pd.cut(df[col], bins=[-np.inf] + qs + [np.inf], labels=labels).astype(str)
    except Exception:
        return pd.Series("mid", index=df.index)

# 价值分层：按 monetary 四分位 -> L1高/L2中高/L3中/L4低
cust["value_tier"] = pd.qcut(cust["monetary_feat"].rank(method="first"), 4,
                             labels=["L4", "L3", "L2", "L1"])
m["value_tier"] = cust.groupby("value_tier").agg(
    n=("CustomerID", "size"),
    monetary=("monetary_feat", "mean"),
    repurchase_rate=("repurchased", "mean"),
).round(2).reset_index().to_dict("records")

# ---- 复购总况 ----
m["repurchase_rate_pct"] = round(float(cust["repurchased"].mean() * 100), 1)
m["churn_pct"] = round(float((1 - cust["repurchased"].mean()) * 100), 1)
m["avg_future_revenue"] = round(float(cust["future_revenue"].mean()), 2)

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=2)
print(json.dumps({k: (v if not isinstance(v, (list, dict)) else f"<{len(v)} 项>") for k, v in m.items()}, ensure_ascii=False))
