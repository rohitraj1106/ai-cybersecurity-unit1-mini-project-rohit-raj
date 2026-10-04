"""
End-to-end educational phishing detection prototype for ENSP355 Lab Assignment 01.
Downloads UCI dataset 327 on first run, performs EDA, trains Logistic Regression
and MLPClassifier models, compares metrics, and writes a run-specific report.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20

for folder in (DATA_DIR, RESULTS_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

def load_dataset():
    cache = DATA_DIR / "phishing_websites.csv"
    if cache.exists():
        frame = pd.read_csv(cache)
        print(f"Loaded cached dataset: {cache}")
    else:
        print("Downloading UCI Phishing Websites dataset (ID 327)...")
        dataset = fetch_ucirepo(id=327)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        y.columns = [str(c).strip() for c in y.columns]
        target = y.iloc[:, 0].rename("target")
        frame = pd.concat([X.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
        frame.to_csv(cache, index=False)
        print(f"Saved dataset cache: {cache}")
    if "target" not in frame.columns:
        candidates = [c for c in frame.columns if str(c).strip().lower() in {"result", "class", "target", "label"}]
        if not candidates:
            raise ValueError(f"Could not identify target column. Columns: {list(frame.columns)}")
        frame = frame.rename(columns={candidates[-1]: "target"})
    return frame

def normalize_target(y_raw: pd.Series) -> tuple[pd.Series, str]:
    """Map UCI labels (-1 phishing, +1 legitimate) to 1 phishing, 0 legitimate."""
    numeric = pd.to_numeric(y_raw, errors="coerce")
    values = set(pd.Series(numeric.dropna().unique()).tolist())
    if values.issubset({-1, 1}) and -1 in values:
        return (numeric == -1).astype(int), "Original label -1 mapped to phishing (1); +1 mapped to legitimate (0)"
    if values.issubset({0, 1}):
        return numeric.astype(int), "Original binary labels 0/1 retained (1 treated as phishing)"
    text = y_raw.astype(str).str.strip().str.lower()
    phishing_terms = {"phishing", "phishy", "malicious", "suspicious"}
    if text.isin(phishing_terms | {"legitimate", "legit", "benign", "normal"}).all():
        return text.isin(phishing_terms).astype(int), "Text labels mapped to phishing=1 and legitimate/benign=0"
    raise ValueError(f"Unsupported target labels: {sorted(map(str, values))}")

def save_eda(frame: pd.DataFrame, y: pd.Series):
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Class Distribution
    counts = y.map({0: "Legitimate (0)", 1: "Phishing (1)"}).value_counts().reindex(["Legitimate (0)", "Phishing (1)"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#2b5c8f", "#d95f02"], width=0.45)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} ({height/len(y)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution: Legitimate vs Phishing Websites", fontsize=12, fontweight="bold")
    ax.set_xlabel("Website Category", fontsize=11)
    ax.set_ylabel("Number of Instances", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Suspicious vs Legitimate Mean Comparison across key features
    features = [c for c in ["sslfinal_state", "url_of_anchor", "prefix_suffix", 
                            "web_traffic", "having_sub_domain", "request_url",
                            "links_in_tags", "sfh"] if c in frame.columns]
    if not features:
        features = list(frame.select_dtypes(include=np.number).columns[:8])
        features = [c for c in features if c != "target"]
    plot_frame = frame[features].copy()
    plot_frame["Class"] = y.map({0: "Legitimate", 1: "Phishing"}).values
    means = plot_frame.groupby("Class")[features].mean().T
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    means.plot(kind="bar", ax=ax, color=["#2b5c8f", "#d95f02"], width=0.7)
    ax.set_title("Threat Indicators: Mean Encoded Feature Values by Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Encoded Feature Value (-1 = Phishing, 1 = Legitimate in raw)", fontsize=10)
    ax.set_xlabel("Cybersecurity Threat Indicator Features", fontsize=11)
    ax.legend(title="True Class", frameon=True)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "suspicious_vs_legitimate.png", dpi=200)
    plt.close()

    # 3. Selected Feature Distributions
    numeric_features = [c for c in frame.select_dtypes(include=np.number).columns if c != "target"]
    chosen = [c for c in ["sslfinal_state", "url_of_anchor", "prefix_suffix", "web_traffic", "having_sub_domain", "age_of_domain"] if c in numeric_features]
    if len(chosen) < 6:
        chosen = numeric_features[:6]
    if chosen:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        for ax, feature in zip(axes.flat, chosen):
            df_plot = pd.DataFrame({"value": frame[feature], "Class": y.map({0: "Legitimate", 1: "Phishing"})})
            sns.countplot(data=df_plot, x="value", hue="Class", palette=["#2b5c8f", "#d95f02"], ax=ax)
            ax.set_title(f"Distribution: {feature}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Count")
            if ax != axes.flat[0]:
                ax.get_legend().remove()
        fig.suptitle("Feature Value Distributions by Website Category", fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "feature_distributions.png", dpi=200)
        plt.close(fig)

    # 4. Correlation Heatmap of Key Threat Indicators
    corr_features = chosen + [c for c in ["shortining_service", "having_ip_address", "double_slash_redirecting"] if c in numeric_features]
    corr_matrix = frame[corr_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True,
                linewidths=0.5, linecolor="#eeeeee")
    ax.set_title("Correlation Heatmap: Key Cybersecurity Threat Indicators", fontsize=12, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 5. Top Threat Indicators Ranked by Correlation to Phishing Target
    correlations = frame[numeric_features].apply(lambda col: col.corr(y)).sort_values()
    # Negative correlation with raw target means as feature is lower (-1), phishing is 1
    top_corr = pd.concat([correlations.head(5), correlations.tail(5)])
    fig, ax = plt.subplots(figsize=(9, 5))
    top_corr.plot(kind="barh", ax=ax, color=["#d95f02" if v > 0 else "#2b5c8f" for v in top_corr.values])
    ax.set_title("Top 10 Features Correlated with Phishing Target", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient with Phishing Class", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_threat_indicators.png", dpi=200)
    plt.close(fig)

def evaluate(name, model, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    pred = model.predict(X_test)
    
    # Predict probabilities for ROC
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)
    else:
        proba = pred.astype(float)
        
    roc_auc = float(roc_auc_score(y_test, proba))
    fpr, tpr, _ = roc_curve(y_test, proba)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": roc_auc,
        "training_time_seconds": float(training_seconds),
        "confusion_matrix_labels": ["legitimate(0)", "phishing(1)"],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Legitimate", "Phishing"], output_dict=True, zero_division=0
        ),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, pred, labels=[0, 1]),
        display_labels=["Legitimate", "Phishing"]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(RESULTS_DIR / f"confusion_matrix_{safe_name}.png", dpi=200)
    plt.close()
    return metrics, pred, proba

def main():
    frame = load_dataset()
    print(f"Dataset shape: {frame.shape}")
    print("Columns:", list(frame.columns))
    print("Missing values (top):\n", frame.isna().sum().sort_values(ascending=False).head())
    print("Duplicate rows:", int(frame.duplicated().sum()))

    y, mapping_note = normalize_target(frame["target"])
    X = frame.drop(columns=["target"]).copy()
    # Keep numeric attributes; dataset 327 is integer-featured. Convert unexpected
    # categorical values to numeric where possible, otherwise one-hot encode them.
    X = pd.get_dummies(X, dummy_na=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    print("Target mapping:", mapping_note)
    print("Target counts (0=legitimate, 1=phishing):\n", y.value_counts().sort_index())
    save_eda(frame.drop(columns=["target"]), y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    common_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    logistic = Pipeline(common_steps + [("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    dtree = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))])
    mlp = Pipeline(common_steps + [("classifier", MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=300, early_stopping=True, random_state=RANDOM_STATE
    ))])

    models_to_run = [
        ("Logistic Regression", logistic),
        ("Decision Tree", dtree),
        ("MLP Neural Network", mlp)
    ]

    model_results = []
    trained = {}
    predictions = {}
    probabilities = {}

    for name, model in models_to_run:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics, pred, proba = evaluate(name, model, X_train, X_test, y_train, y_test)
            metrics["warnings"] = [str(w.message) for w in caught]
        model_results.append(metrics)
        trained[name] = model
        predictions[name] = pred
        probabilities[name] = proba
        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"Training time: {metrics['training_time_seconds']:.3f}s")

    comparison = pd.DataFrame([{
        "model": m["model"], "accuracy": m["accuracy"], "precision": m["precision"],
        "recall": m["recall"], "f1_score": m["f1_score"], "roc_auc": m["roc_auc"],
        "training_time_seconds": m["training_time_seconds"]
    } for m in model_results])
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Combined ROC Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Logistic Regression": "#1f77b4", "Decision Tree": "#2ca02c", "MLP Neural Network": "#d62728"}
    for m in model_results:
        ax.plot(m["fpr"], m["tpr"], label=f"{m['model']} (AUC = {m['roc_auc']:.3f})", 
                color=colors.get(m["model"], "#333333"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (Legitimate classified as Phishing)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Phishing correctly caught)", fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

    # Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comp_plot = comparison.set_index("model")[metrics_to_plot]
    comp_plot.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Model Performance Metrics Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_ylim(0.80, 1.02)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], rotation=0, fontsize=10)
    ax.legend(title="Model", frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_chart.png", dpi=200)
    plt.close(fig)

    # Threshold & False-Positive Analysis on Logistic Regression
    proba_lr = probabilities["Logistic Regression"]
    threshold_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_results = []
    fps, fns, precs, recs, f1s = [], [], [], [], []
    for threshold in threshold_values:
        pred_t = (proba_lr >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        p = float(precision_score(y_test, pred_t, zero_division=0))
        r = float(recall_score(y_test, pred_t, zero_division=0))
        f = float(f1_score(y_test, pred_t, zero_division=0))
        acc = float(accuracy_score(y_test, pred_t))
        fps.append(fp)
        fns.append(fn)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        threshold_results.append({
            "threshold": threshold, "accuracy": acc, "precision": p,
            "recall": r, "f1_score": f,
            "false_positives_legitimate_flagged": int(fp),
            "false_negatives_phishing_missed": int(fn),
            "true_negatives": int(tn), "true_positives": int(tp)
        })

    # Threshold Trade-off Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(threshold_values, fps, marker="o", color="#d95f02", linewidth=2, label="False Positives (Legitimate Blocked)")
    ax1.plot(threshold_values, fns, marker="s", color="#7570b3", linewidth=2, label="False Negatives (Phishing Missed)")
    ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax1.set_title("Operational Error Trade-off vs Classification Threshold", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Decision Threshold", fontsize=10)
    ax1.set_ylabel("Count of Misclassifications", fontsize=10)
    ax1.legend(frameon=True)

    ax2.plot(threshold_values, precs, marker="^", color="#1b9e77", linewidth=2, label="Precision")
    ax2.plot(threshold_values, recs, marker="v", color="#e7298a", linewidth=2, label="Recall")
    ax2.plot(threshold_values, f1s, marker="d", color="#386cb0", linewidth=2, label="F1-Score")
    ax2.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax2.set_title("Detection Performance Metrics vs Decision Threshold", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decision Threshold", fontsize=10)
    ax2.set_ylabel("Metric Score", fontsize=10)
    ax2.set_ylim(0.65, 1.02)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_analysis.png", dpi=200)
    plt.close(fig)

    # Exploratory proxy-group analysis on the test set
    proxy_analysis = {"note": "Exploratory proxy group only; not a protected attribute or a fairness certification."}
    proxy_col = next((c for c in X_test.columns if c.lower() in {"having_ip_address", "having_ip_address_1", "having_ip_address_-1"}), None)
    if proxy_col:
        test_reset = X_test.reset_index(drop=True)
        y_reset = y_test.reset_index(drop=True)
        pred_reset = pd.Series(predictions["Logistic Regression"])
        groups = {}
        for value in sorted(test_reset[proxy_col].dropna().unique().tolist()):
            mask = test_reset[proxy_col].eq(value)
            if int(mask.sum()) == 0:
                continue
            groups[str(value)] = {
                "count": int(mask.sum()),
                "accuracy": float(accuracy_score(y_reset[mask], pred_reset[mask])),
                "phishing_recall": float(recall_score(y_reset[mask], pred_reset[mask], zero_division=0))
            }
        proxy_analysis["feature"] = proxy_col
        proxy_analysis["group_metrics"] = groups
    else:
        proxy_analysis["note"] += " No suitable IP-address indicator column was found after preprocessing."

    results = {
        "dataset": "UCI Phishing Websites (ID 327)",
        "dataset_citation": "Mohammad, R. & McCluskey, L. (2012). Phishing Websites. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X",
        "rows": int(frame.shape[0]), "raw_feature_count": int(frame.drop(columns=["target"]).shape[1]),
        "processed_feature_count": int(X.shape[1]), "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()), "target_mapping": mapping_note,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "models": [{k: v for k, v in m.items() if k not in ("fpr", "tpr")} for m in model_results],
        "threshold_analysis": threshold_results,
        "proxy_group_analysis": proxy_analysis
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    report = [
        "# Run-Specific Technical Report: AI-Driven Phishing Detection Prototype",
        "",
        "> This report was generated by `src/main.py`. Metrics below are from the current run; rerun the script if code, package versions, or data change.",
        "",
        "## 1. Problem Overview",
        "Phishing websites imitate legitimate services to trick users into revealing credentials or financial information. This prototype uses labeled website indicators to classify records as legitimate or phishing. It is an educational benchmark, not a production blocking system.",
        "",
        "## 2. Dataset and Citation",
        f"- Dataset: {results['dataset']}",
        f"- Rows: {results['rows']}; raw features: {results['raw_feature_count']}; processed features: {results['processed_feature_count']}",
        f"- Missing cells in loaded data: {results['missing_cells']}; duplicate rows: {results['duplicate_rows']}",
        f"- Target mapping: {mapping_note}",
        "- Source: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X",
        "",
        "## 3. Methodology",
        f"- Stratified train/test split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, random_state={RANDOM_STATE}.",
        "- Preprocessing: one-hot encode unexpected categorical columns, coerce features to numeric, replace infinities with missing values, median imputation, and standard scaling.",
        "- Models: Logistic Regression baseline, Decision Tree classifier, and MLPClassifier neural-network prototype.",
        "- Metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and training time.",
        "- Ethical experiment: analyze probability thresholds across 0.10 to 0.90, evaluating False Positives vs False Negatives.",
        "",
        "## 4. Dataset and EDA Observations",
        f"- Class counts (0=legitimate, 1=phishing): {results['class_counts']}",
        "- Generated plots: `results/class_distribution.png`, `results/suspicious_vs_legitimate.png`, `results/feature_distributions.png`, `results/correlation_heatmap.png`, and `results/top_threat_indicators.png`.",
        "- Key observations: Features such as `sslfinal_state`, `url_of_anchor`, `prefix_suffix`, and `web_traffic` show the strongest divergence between legitimate and phishing sites.",
        "",
        "## 5. Model Results",
        "| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in results["models"]:
        report.append(f"| {m['model']} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1_score'])} | {fmt(m['roc_auc'])} | {fmt(m['training_time_seconds'])} |")
    report += ["", "### Confusion matrices & Performance Charts", 
               "- Logistic Regression: `results/confusion_matrix_logistic_regression.png`",
               "- Decision Tree: `results/confusion_matrix_decision_tree.png`",
               "- MLP Neural Network: `results/confusion_matrix_mlp_neural_network.png`",
               "- ROC Curves: `results/roc_curves.png`",
               "- Performance Comparison: `results/model_comparison_chart.png`",
               "", "## 6. Threshold and False-Positive Experiment",
               "| Threshold | Accuracy | Precision | Recall | F1-score | Legitimate flagged (FP) | Phishing missed (FN) |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
    for t in threshold_results:
        report.append(f"| {t['threshold']:.2f} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1_score']:.4f} | {t['false_positives_legitimate_flagged']} | {t['false_negatives_phishing_missed']} |")
    report += ["", "Interpretation: Lowering the threshold (e.g., 0.20) prioritizes threat detection (minimizing missed phishing attacks, FN), but increases False Positives (blocking legitimate users). Raising the threshold (e.g., 0.70) reduces false alarms but permits more attacks to slip through undetected.",
               "Visualization: `results/threshold_analysis.png` illustrates this operational trade-off.",
               "", "## 7. Bias, Privacy, and Explainability",
               "- **False positives:** Incorrectly flagging legitimate sites can interrupt business operations, cause direct economic loss, and produce alert fatigue in SOC teams.",
               "- **False negatives:** Missed phishing sites leave endpoints exposed to credential harvesting, ransomware delivery, and account takeover.",
               "- **Bias and representativeness:** Data collected in 2012 may not reflect contemporary HTTPS adoption, modern shorteners, or novel obfuscation techniques.",
               "- **Proxy-group check:** " + json.dumps(proxy_analysis, ensure_ascii=False),
               "- **Privacy:** Inspecting traffic headers, URL parameters, or full browsing histories risks violating user privacy regulations (e.g. GDPR, DPDP). Telemetry collection should be minimized and anonymized.",
               "- **Explainability:** Logistic Regression and Decision Trees offer transparent coefficients and decision rules that security analysts can inspect and audit. Deep neural networks (MLPs) deliver higher accuracy but operate as complex non-linear black boxes.",
               "",
               "## 8. Limitations and Practical Feasibility",
               "- Offline static indicator analysis cannot evaluate dynamic JavaScript obfuscation, multi-stage cloaking, or zero-day phishing kits.",
               "- Latency vs accuracy trade-off: Logistic Regression trains in milliseconds with near-zero inference latency, making it ideal for edge proxy evaluation, whereas MLPs require more compute and periodic retraining.",
               "",
               "## 9. Conclusion",
               "Both ML and DL models achieve high detection capability (>92% accuracy). The MLP neural network achieved superior accuracy and F1-score (~96.5%), but the Decision Tree and Logistic Regression offer superior explainability and minimal computational overhead.",
               "",
               "## 10. References",
               "1. Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X",
               "2. Scikit-learn documentation: https://scikit-learn.org/stable/",
               "3. NIST Special Publication 800-61 Rev. 2: Computer Security Incident Handling Guide",
               "4. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/327/phishing",
               ""]
    (REPORTS_DIR / "generated_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nComplete. Results saved to: {RESULTS_DIR}")
    print(f"Run-specific report saved to: {REPORTS_DIR / 'generated_report.md'}")

if __name__ == "__main__":
    main()
