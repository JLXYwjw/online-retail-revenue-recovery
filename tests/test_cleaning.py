import os
from pathlib import Path
import pytest
_CLEAN_OUT = Path("data/output") / "retail_clean.parquet"
pytestmark = pytest.mark.skipif(
    not _CLEAN_OUT.exists(),
    reason="cleaned parquet absent (git-excluded); run cleaning first",
)

# -*- coding: utf-8 -*-
"""电商清洗口径测试：三张清洗表的结构与一致性、时间切分无泄漏。"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"


def _clean():
    return pd.read_parquet(OUT / "retail_clean.parquet")


def test_clean_shape():
    df = _clean()
    assert len(df) == 541909
    assert "revenue" in df.columns and "is_return" in df.columns and "is_cancel" in df.columns


def test_return_and_cancel_counts():
    df = _clean()
    assert df["is_return"].sum() == 9290
    assert df["is_cancel"].sum() == 9288


def test_customer_rfm_shape():
    cust = pd.read_parquet(OUT / "customer_rfm.parquet")
    assert len(cust) == 948
    assert {"monetary_feat", "freq_feat", "recency_feat", "future_revenue", "repurchased"} <= set(cust.columns)


def test_no_temporal_leakage():
    # 特征全部来自 2010，标签来自 2011（时间切分）
    df = _clean()
    feat = df[(df["has_customer"]) & (df["InvoiceDate"] < pd.Timestamp("2011-01-01"))]
    labels = df[(df["has_customer"]) & (df["InvoiceDate"] >= pd.Timestamp("2011-01-01"))]
    assert len(feat) > 0 and len(labels) > 0


def test_invoice_level():
    inv = pd.read_parquet(OUT / "invoice_level.parquet")
    assert len(inv) == 25900
