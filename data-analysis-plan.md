# Online Retail: Revenue Reconciliation & Customer Value — Business Plan

## 1. Business Objectives (driven by the business)
An online retailer (UK, UCI Online Retail real transaction data) needs to answer:
- **Revenue reconciliation**: booked "revenue to recognize" vs actual net revenue — how much and where is the revenue leakage from returns/cancellations/negative unit prices.
- **Customer value & churn**: of the 4,372 customers, which are high-value (prioritize) and which are churning — put limited operations/marketing budget into the most valuable customers.
- **Delivery**: executable revenue-recovery (reduce return loss / optimize assortment) and customer-operations (retain high-value customers) recommendations.

## 2. Data
- Source: `datasets/ecommerce/Online_Retail.csv` (UCI classic real data), 541,909 rows / 8 columns.
- Fields: InvoiceNo (C prefix = cancellation), StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
- Time: 2010-01-12 → 2011-12-10.
- Known dirty features: CustomerID missing 24.9%, Description missing 1,454, negative Quantity/UnitPrice (returns/refunds), 9,288 cancellation lines.

## 3. Cleaning & Preprocessing
- Drop/mark rows without CustomerID (anonymous transactions; revenue measured two ways: with/without).
- Fill/drop missing Description.
- Derive: revenue = Quantity × UnitPrice; invoice-level aggregation; flag fields (cancellation, return lines).
- Customer-level features (RFM): total amount, purchase count, days since last purchase, purchase interval, return amount.
- Customer split by time: first 12 months train / later observation window validate (customer repurchase prediction, time split is valid).

## 4. Metrics
- Revenue reconciliation: total revenue (>0) $10.67M; returns/cancellations (<0) -$0.92M (1.7%); net revenue; cancellation amount.
- Customer: customer count, per-customer GMV, RFM tiers, top N% customer contribution share (concentration).
- Repurchase/churn: repurchase rate in observation window, churned-customer share.

## 5. Modeling (used only where the business needs it; explain principle/why/how)
- Objective A (customer value regression): predict customer "future spend" (CLV) from RFM features, to rank customers for operations.
- Objective B (repurchase classification): predict whether a customer buys again after the observation window (churn warning) — time split, no leakage.
- Methods: baseline linear regression / logistic regression + tree models (RF/GBM); DL judgment: ~4,400 customers, no sequence advantage, DL adds little and is hard to explain — DL not used, reason stated.
- Metrics: regression RMSE/R², classification AUC/PR; honest limits on sample size and time window.

## 6. Visualizations (each chart Insight/Action)
- Revenue reconciliation waterfall/gap: total revenue → returns → net.
- Monthly revenue and return trend.
- Customer contribution concentration (Pareto: top X% of customers contribute Y%).
- RFM / customer value tier bars.
- ML feature importance / CLV predicted vs actual.
- Repurchase classification ROC/AUC + high-risk customer profile.

## 7. Recommendations (landing)
- Revenue recovery: optimize return-prone SKUs/countries, cancellation flow.
- Customer operations: tiered operations for high-value customers, churn-warning triggered repurchase outreach.
- Next step: line-level reconciliation needs payment/refund flow, listed as an external data gap.

## 8. Deliverables
- Bilingual HTML reports (third palette: navy + coral), each chart Insight/Action.
- Repo: scripts/tests/data/output + README + requirements; local git committed in small steps.
