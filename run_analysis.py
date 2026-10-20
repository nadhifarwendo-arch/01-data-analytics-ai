"""Reproducible synthetic business analytics + AI demo inspired by the transcript.

No real company data is used. Run: python run_analysis.py
"""
from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
RNG = np.random.default_rng(77)


def create_synthetic_data():
    """Create a small, repeatable mock business dataset (120 customers, 24 products)."""
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    start = pd.Timestamp("2024-07-01")
    end = pd.Timestamp("2026-06-30")

    n_customers, n_products, n_invoices = 120, 24, 4000
    segments = np.array(["Retail", "Wholesale", "Government", "SME"])
    regions = np.array(["Bengkulu", "Sumatera Selatan", "Jambi", "Lampung"])
    customers = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(1, n_customers + 1)],
        "segment": RNG.choice(segments, n_customers, p=[0.42, 0.28, 0.08, 0.22]),
        "region": RNG.choice(regions, n_customers),
        "synthetic_risk_score": RNG.uniform(0.05, 0.75, n_customers).round(3),
    })

    category_names = ["Hardware", "Office Supplies", "Electronics", "Furniture"]
    products_rows = []
    for i in range(n_products):
        base_price = float(RNG.uniform(35_000, 1_500_000))
        margin = float(RNG.uniform(0.12, 0.42))
        category = category_names[i // 6]
        # Deliberate inconsistent category casing is cleaned later.
        cat_variant = category.lower() if i % 5 == 0 else (category.upper() if i % 7 == 0 else category)
        products_rows.append({
            "product_id": f"P{i+1:03d}",
            "product_name": f"Produk {i+1:02d}",
            "category": cat_variant,
            "base_price": round(base_price, 2),
            "base_cost": round(base_price * (1 - margin), 2),
        })
    products = pd.DataFrame(products_rows)

    invoice_dates = pd.to_datetime(RNG.integers(start.value // 10**9, end.value // 10**9, n_invoices), unit="s").normalize()
    invoice_dates = pd.Series(invoice_dates).sort_values(ignore_index=True)
    cust_ids = RNG.choice(customers["customer_id"], n_invoices)
    cust_risk = customers.set_index("customer_id")["synthetic_risk_score"].to_dict()
    methods = np.array(["Transfer", "Cash", "Card", "E-Wallet"])
    invoice_rows, line_rows = [], []

    for i, inv_date in enumerate(invoice_dates):
        invoice_id = f"INV{i+1:05d}"
        customer_id = str(cust_ids[i])
        risk = cust_risk[customer_id]
        month_index = max(0, (inv_date.year - 2024) * 12 + inv_date.month - 7)
        discount = float(RNG.choice([0.0, 0.02, 0.05, 0.10, 0.15], p=[0.24, 0.22, 0.26, 0.20, 0.08]))
        # Six distinct products per invoice -> approximately 24,000 invoice-line records.
        selected = RNG.choice(n_products, 6, replace=False)
        net_total, cost_total = 0.0, 0.0
        for j, product_index in enumerate(selected, start=1):
            p = products.iloc[int(product_index)]
            qty = int(RNG.integers(1, 5))
            # Sales price grows slowly; operating cost grows faster to create a useful trend case.
            sale_price = float(p["base_price"] * (1 + 0.003 * month_index))
            unit_cost = float(p["base_cost"] * (1 + 0.010 * month_index))
            net_line = qty * sale_price * (1 - discount)
            cost_line = qty * unit_cost
            net_total += net_line
            cost_total += cost_line
            line_rows.append({
                "invoice_id": invoice_id,
                "line_no": j,
                "product_id": p["product_id"],
                "quantity": qty,
                "discount_rate": discount,
            })

        late_probability = min(0.78, 0.08 + 0.46 * risk + 0.22 * discount + 0.005 * month_index)
        is_unpaid = bool(RNG.random() < late_probability * 0.35)
        is_late = bool(RNG.random() < late_probability)
        if is_unpaid:
            payment_date = pd.NaT
            payment_status = "Unpaid"
        else:
            days_to_pay = int(RNG.integers(31, 91)) if is_late else int(RNG.integers(3, 31))
            payment_date = inv_date + pd.Timedelta(days=days_to_pay)
            payment_status = "Paid"
        method = str(RNG.choice(methods))
        if RNG.random() < 0.025:
            method = None
        invoice_rows.append({
            "invoice_id": invoice_id,
            "customer_id": customer_id,
            "invoice_date": inv_date.strftime("%Y-%m-%d"),
            "due_date": (inv_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "payment_date": payment_date.strftime("%Y-%m-%d") if pd.notna(payment_date) else None,
            "payment_status": payment_status,
            "payment_method": method,
            "discount_rate": discount,
        })

    invoices = pd.DataFrame(invoice_rows)
    lines = pd.DataFrame(line_rows)
    # Inject a few known data-quality issues to demonstrate cleaning.
    invoices = pd.concat([invoices, invoices.iloc[[15]]], ignore_index=True)
    if len(lines) > 60:
        lines = pd.concat([lines, lines.iloc[[23, 41, 59]]], ignore_index=True)
        lines.loc[100, "quantity"] = np.nan
    customers.to_csv(DATA / "customers.csv", index=False)
    products.to_csv(DATA / "products_raw.csv", index=False)
    invoices.to_csv(DATA / "invoices_raw.csv", index=False)
    lines.to_csv(DATA / "invoice_lines_raw.csv", index=False)
    print("Data sintetis disiapkan: 120 pelanggan, 24 produk, sekitar 4.000 invoice.")


def main():
    if not (DATA / "invoices_raw.csv").exists():
        create_synthetic_data()
    OUT.mkdir(exist_ok=True)

    invoices = pd.read_csv(DATA / "invoices_raw.csv")
    lines = pd.read_csv(DATA / "invoice_lines_raw.csv")
    customers = pd.read_csv(DATA / "customers.csv")
    products = pd.read_csv(DATA / "products_raw.csv")

    # Cleaning: remove repeated keys, standardize category labels, and handle nulls.
    invoice_duplicates = int(invoices.duplicated("invoice_id").sum())
    line_duplicates = int(lines.duplicated(["invoice_id", "line_no"]).sum())
    invoices = invoices.drop_duplicates("invoice_id", keep="first").copy()
    lines = lines.drop_duplicates(["invoice_id", "line_no"], keep="first").copy()
    lines["quantity"] = pd.to_numeric(lines["quantity"], errors="coerce")
    lines["quantity"] = lines["quantity"].fillna(lines["quantity"].median()).clip(lower=1)
    lines["quantity"] = lines["quantity"].round().astype(int)
    invoices["payment_method"] = invoices["payment_method"].fillna("Tidak diketahui").str.strip().str.title()
    products["category"] = products["category"].astype(str).str.strip().str.title()
    for col in ["invoice_date", "due_date", "payment_date"]:
        invoices[col] = pd.to_datetime(invoices[col], errors="coerce")
    invoices["discount_rate"] = pd.to_numeric(invoices["discount_rate"], errors="coerce").fillna(0).clip(0, 0.8)

    # Calculate financial values at line-level first, then aggregate to invoice grain.
    line_data = lines.merge(products, on="product_id", how="left", validate="many_to_one")
    month_index = ((pd.to_datetime(invoices["invoice_date"]).dt.year - 2024) * 12 +
                   pd.to_datetime(invoices["invoice_date"]).dt.month - 7).clip(lower=0)
    trend = invoices.set_index("invoice_id")["invoice_date"].to_dict()
    line_data["invoice_date"] = line_data["invoice_id"].map(trend)
    line_data["month_index"] = ((line_data["invoice_date"].dt.year - 2024) * 12 +
                                line_data["invoice_date"].dt.month - 7).clip(lower=0)
    line_data["unit_price"] = line_data["base_price"] * (1 + 0.003 * line_data["month_index"])
    line_data["unit_cost"] = line_data["base_cost"] * (1 + 0.010 * line_data["month_index"])
    line_data["net_sales"] = line_data["quantity"] * line_data["unit_price"] * (1 - line_data["discount_rate"])
    line_data["cost"] = line_data["quantity"] * line_data["unit_cost"]
    line_data["gross_profit"] = line_data["net_sales"] - line_data["cost"]
    invoice_lines = line_data.groupby("invoice_id", as_index=False).agg(
        invoice_total=("net_sales", "sum"), invoice_cost=("cost", "sum"),
        gross_profit=("gross_profit", "sum"), line_count=("line_no", "count"),
        avg_discount=("discount_rate", "mean"),
    )
    # Validate many-to-one joins to reduce risk of accidental row multiplication.
    enriched = invoices.merge(invoice_lines, on="invoice_id", how="left", validate="one_to_one")
    enriched = enriched.merge(customers[["customer_id", "segment", "region", "synthetic_risk_score"]],
                              on="customer_id", how="left", validate="many_to_one")
    enriched["receivable"] = np.where(enriched["payment_status"].eq("Unpaid"), enriched["invoice_total"], 0.0)
    enriched["late_or_unpaid"] = (
        enriched["payment_status"].eq("Unpaid") |
        (enriched["payment_date"].notna() & ((enriched["payment_date"] - enriched["due_date"]).dt.days > 0))
    ).astype(int)
    enriched["month"] = enriched["invoice_date"].dt.to_period("M").astype(str)
    enriched = enriched.sort_values(["customer_id", "invoice_date", "invoice_id"]).reset_index(drop=True)
    enriched["customer_late_rate_history"] = enriched.groupby("customer_id")["late_or_unpaid"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    enriched["customer_late_rate_history"] = enriched["customer_late_rate_history"].fillna(enriched["late_or_unpaid"].mean())
    # Evaluate on the latest 20% of dates, rather than randomly mixing future and past records.
    enriched = enriched.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
    enriched.to_csv(OUT / "invoice_analysis_clean.csv", index=False)

    # Descriptive analytics: what happened?
    monthly = enriched.groupby("month", as_index=False).agg(
        revenue=("invoice_total", "sum"), gross_profit=("gross_profit", "sum"),
        receivables=("receivable", "sum"), invoice_count=("invoice_id", "nunique"),
        overdue_rate=("late_or_unpaid", "mean"),
    )
    monthly.to_csv(OUT / "monthly_kpis.csv", index=False)

    # Diagnostic analytics: where does the problem appear?
    customer_diagnostics = enriched.groupby(["customer_id", "region", "segment"], as_index=False).agg(
        invoice_count=("invoice_id", "nunique"), revenue=("invoice_total", "sum"),
        receivables=("receivable", "sum"), late_rate=("late_or_unpaid", "mean"),
    ).sort_values(["receivables", "late_rate"], ascending=False)
    customer_diagnostics.to_csv(OUT / "customer_diagnostics.csv", index=False)

    # Predictive analytics: predict late/unpaid label from information available at invoice time.
    # Payment date/status is deliberately excluded from model features to avoid target leakage.
    numeric = ["invoice_total", "avg_discount", "line_count", "synthetic_risk_score", "customer_late_rate_history"]
    categorical = ["payment_method", "segment", "region"]
    X, y = enriched[numeric + categorical].copy(), enriched["late_or_unpaid"]
    split_at = int(len(enriched) * 0.8)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Label sintetis tidak memiliki dua kelas; jalankan ulang dengan seed berbeda.")
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    model = Pipeline([("preprocess", preprocessor),
                      ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pred, zero_division=0)
    try:
        auc_text = f"ROC-AUC: {roc_auc_score(y_test, probability):.3f}\n"
    except ValueError:
        auc_text = "ROC-AUC tidak tersedia untuk pembagian kelas ini.\n"
    (OUT / "model_evaluation.txt").write_text(
        "MODEL PREDIKSI TELAT BAYAR — DATA SINTETIS\n"
        "Fitur tidak mencakup payment_date atau payment_status sebagai prediktor.\n\n" + auc_text +
        "Classification report:\n" + report + "\nConfusion matrix:\n" + str(confusion_matrix(y_test, pred)) + "\n",
        encoding="utf-8"
    )
    scored = enriched.copy()
    scored["predicted_late_probability"] = model.predict_proba(X)[:, 1]
    action_list = scored[scored["payment_status"].eq("Unpaid")].sort_values(
        ["predicted_late_probability", "receivable"], ascending=False
    )[["invoice_id", "customer_id", "region", "invoice_total", "receivable", "predicted_late_probability"]]
    action_list.head(30).to_csv(OUT / "priority_follow_up.csv", index=False)

    # Visualizations: keep them simple, with labels and legible scales.
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["month"], monthly["revenue"] / 1_000_000, marker="o", label="Revenue (juta Rp)")
    plt.plot(monthly["month"], monthly["gross_profit"] / 1_000_000, marker="o", label="Gross profit (juta Rp)")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Juta rupiah")
    plt.title("Tren Revenue dan Gross Profit — Data Sintetis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "trend_revenue_profit.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    top = customer_diagnostics.head(10).sort_values("receivables")
    plt.barh(top["customer_id"], top["receivables"] / 1_000_000)
    plt.xlabel("Piutang (juta Rp)")
    plt.title("10 Pelanggan dengan Piutang Terbesar")
    plt.tight_layout()
    plt.savefig(OUT / "top_receivables_customers.png", dpi=150)
    plt.close()

    print(f"Invoice unik: {len(enriched):,}; duplikat header dibuang: {invoice_duplicates}; duplikat baris dibuang: {line_duplicates}.")
    print(f"Data invoice-line setelah pembersihan: {len(lines):,} baris.")
    print(f"Baris akhir tersimpan di: {OUT}")
    print("\nRingkasan bulanan terbaru:")
    print(monthly.tail(3).to_string(index=False))
    print("\n" + auc_text + report)
    print("Catatan: seluruh data dibuat secara sintetis untuk latihan, bukan laporan keuangan perusahaan nyata.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
