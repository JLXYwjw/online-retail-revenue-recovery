# -*- coding: utf-8 -*-
"""报告数字一致性：从 metrics/ml 重算关键数字，断言 HTML 含同数字。"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
HTML = ROOT / "report" / "电商收入对账与客户价值分析.html"


def _html():
    return HTML.read_text(encoding="utf-8")


def _m():
    return json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def test_net_return_match_html():
    m = _m()
    assert "$9.75M" in _html()
    assert abs(m["net_revenue"] / 1e6 - 9.75) < 0.02
    assert "-$0.92M" in _html() and abs(-m["return_leakage"] / 1e6 - 0.92) < 0.02


def test_concentration_match_html():
    m = _m()
    assert abs(m["top10pct_share"] - 60.0) < 0.1
    assert "60%" in _html() and "48.5%" in _html() and "73.8%" in _html()


def test_repurchase_rate_match_html():
    m = _m()
    assert abs(m["repurchase_rate_pct"] - 86.5) < 0.1
    assert "86.5%" in _html()


def test_sep2011_return_match_html():
    m = _m()
    r = next(x for x in m["monthly_return"] if x["ym"] == "2011-09")
    assert abs(r["return_amt"] / 1000 - 212.9) < 2
    assert "212.9K" in _html()


def test_ml_numbers_match_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    gbm_auc = ml["classification"]["gbm_cls"]["auc"]
    assert abs(gbm_auc - 0.627) < 0.01
    assert "0.63" in _html()
    lin_r2 = ml["regression"]["linear"]["r2"]
    assert abs(lin_r2 - 0.318) < 0.01
    assert "0.32" in _html() or "0.318" in _html()
