# -*- coding: utf-8 -*-
"""探查 Online_Retail.csv：结构、真实脏特征、时间范围、取消/退款线索。"""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "Online_Retail.csv"
df = pd.read_csv(CSV, encoding="latin1")

print("shape:", df.shape)
print("columns:", list(df.columns))
print("\ndtypes:\n", df.dtypes.to_string())
print("\n列缺失数:\n", df.isna().sum().to_string())

# 时间范围
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
print("\nInvoiceDate range:", df["InvoiceDate"].min(), "->", df["InvoiceDate"].max())
print("日期解析失败:", int(df["InvoiceDate"].isna().sum()))

# 发票号前缀(取消单 C 开头)
inv = df["InvoiceNo"].astype(str)
print("\n发票号前缀分布 top:")
print(inv.str[0].value_counts().head().to_string())
print("取消单(C开头)行数:", int(inv.str.startswith("C").sum()))

print("\nCountry top10:\n", df["Country"].value_counts().head(10).to_string())

print("\n单价描述:\n", df["UnitPrice"].describe().to_string())
print("单价<=0 行:", int((df["UnitPrice"] <= 0).sum()), "单价<0:", int((df["UnitPrice"] < 0).sum()))
print("数量描述:\n", df["Quantity"].describe().to_string())
print("数量<=0 行:", int((df["Quantity"] <= 0).sum()))

print("\nCustomerID 缺失数:", int(df["CustomerID"].isna().sum()), f"({df['CustomerID'].isna().mean()*100:.1f}%)")
print("StockCode top10:\n", df["StockCode"].value_counts().head(10).to_string())
print("唯一 StockCode:", df["StockCode"].nunique(), "唯一 CustomerID:", df["CustomerID"].nunique(), "唯一发票:", df["InvoiceNo"].nunique())

# 关键口径：净营收(负数量=退货/取消)、取消单金额
df["revenue"] = df["Quantity"] * df["UnitPrice"]
pos = df[df["revenue"] > 0]["revenue"]
neg = df[df["revenue"] < 0]["revenue"]
print("\n营收>0 合计:", round(float(pos.sum()), 2), "营收<0(退货/取消)合计:", round(float(neg.sum()), 2))
print("营收<0 行数:", len(neg), f"({len(neg)/len(df)*100:.1f}%)")
cancel = df[df["InvoiceNo"].astype(str).str.startswith("C")]
print("取消单(C) 金额:", round(float(cancel["revenue"].sum()), 2), "取消单行:", len(cancel))
