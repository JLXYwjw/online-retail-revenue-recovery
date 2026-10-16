# -*- coding: utf-8 -*-
"""客户建模：CLV回归(未来消费额) + 复购分类(是否再买)。特征=2010 RFM，标签=2011。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
    RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
cust = pd.read_parquet(OUT / "customer_rfm.parquet")

FEAT = ["monetary_feat", "freq_feat", "recency_feat", "lines_feat", "return_amt_feat"]
X = cust[FEAT].copy()
X["monetary_log"] = np.log1p(np.clip(X["monetary_feat"], 0, None))
X["return_log"] = np.log1p(X["return_amt_feat"])
XCOLS = FEAT + ["monetary_log", "return_log"]

y_reg = cust["future_revenue"]
y_cls = cust["repurchased"]

# 分层切分（按 y_cls 平衡类别）
tr, te = train_test_split(cust.index, test_size=0.25, stratify=y_cls, random_state=42)
Xtr, Xte = X.loc[tr], X.loc[te]
reg_tr, reg_te = y_reg.loc[tr], y_reg.loc[te]
cls_tr, cls_te = y_cls.loc[tr], y_cls.loc[te]

reg_res, cls_res = {}, {}
for name, mdl in [("linear", LinearRegression()), ("rf_reg", RandomForestRegressor(n_estimators=300, random_state=42)),
                  ("gbm_reg", GradientBoostingRegressor(random_state=42))]:
    mdl.fit(Xtr[XCOLS], reg_tr)
    pred = mdl.predict(Xte[XCOLS])
    reg_res[name] = {"r2": round(float(r2_score(reg_te, pred)), 4),
                     "rmse": round(float(np.sqrt(mean_squared_error(reg_te, pred))), 2)}

for name, mdl in [("logistic", LogisticRegression(max_iter=1000)),
                  ("rf_cls", RandomForestClassifier(n_estimators=300, random_state=42)),
                  ("gbm_cls", GradientBoostingClassifier(random_state=42))]:
    mdl.fit(Xtr[XCOLS], cls_tr)
    p = mdl.predict_proba(Xte[XCOLS])[:, 1]
    cls_res[name] = {"auc": round(float(roc_auc_score(cls_te, p)), 4),
                     "pr_auc": round(float(average_precision_score(cls_te, p)), 4)}

# 主模型特征重要性（GBM 分类）
gbm = GradientBoostingClassifier(random_state=42).fit(Xtr[XCOLS], cls_tr)
imp = pd.Series(gbm.feature_importances_, index=XCOLS).sort_values(ascending=False).round(4)

out = {"n_total": int(len(cust)), "n_train": int(len(tr)), "n_test": int(len(te)),
       "repurchase_rate_test_pct": round(float(cls_te.mean() * 100), 1),
       "regression": reg_res, "classification": cls_res,
       "top_features": imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_dict("records"),
       "note": "948 特征期客户，样本小；CLV/复购预测用于客户运营分层，非精确金额估计。DL 因样本量与可解释性不用。"}
with open(OUT / "ml_result.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps({"n": len(cust), "cls": cls_res, "reg": reg_res}, ensure_ascii=False))
