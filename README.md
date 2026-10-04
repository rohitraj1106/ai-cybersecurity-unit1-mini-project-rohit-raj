# AI-Driven Cyber Threat Awareness and Detection Prototype

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF-red.svg)](https://www.reportlab.com/)

**Course:** AI in Cyber Security Lab (ENSP355)  
**Lab Assignment:** 01 — Foundations of AI in Cyber Security  
**Student Name:** Rohit Raj  
**University:** K.R. Mangalam University  
**Faculty:** Monika Khatkar         
**Weightage:** 20% of Lab Assessment (100 Marks Rubric)  

---

## Executive Summary
Modern enterprise environments rely on complex digital infrastructure vulnerable to phishing campaigns, credential harvesting, malware delivery, and automated cyber attacks. Traditional security tools based on static signatures or static domain blacklists are incapable of stopping adaptive adversaries utilizing dynamic domain generation algorithms (DGAs) and zero-hour obfuscation.

This project delivers an **end-to-end, production-grade cybersecurity threat detection prototype** trained on 11,055 website instances from the UCI Machine Learning Repository (Dataset ID 327). Across all 5 assigned tasks, it implements:
1. **Workspace Bootstrap & Environment Setup** (Virtual environment, dependencies, verified test runners).
2. **Cyber Security Threat Landscape Analysis** (EDA, class balance, threat indicators, correlation heatmap, and ranking).
3. **Supervised Machine Learning Threat Classification** (Standardized preprocessing, Logistic Regression baseline, and Decision Tree classifier).
4. **Deep Learning Prototype Implementation** (Multi-Layer Perceptron neural network, multi-metric benchmarking, ROC curves, and SOC latency feasibility).
5. **Ethical Reflection & Operational AI Challenges** (Granular threshold simulation from 0.10 to 0.90, false-positive user lockout analysis, proxy bias examination, and GDPR/DPDP privacy reviews).

---

## Project Structure
```text
ai-cybersecurity-unit1-mini-project/
├── .gitignore                          # Ignores venvs, cache, and temp files; tracks figures & deliverables
├── README.md                           # Complete project guide and execution instructions
├── requirements.txt                    # Project dependencies (scikit-learn, pandas, jupyter, reportlab, etc.)
├── data/
│   ├── .gitkeep
│   └── phishing_websites.csv           # Cached UCI benchmark dataset (11,055 rows, 31 columns)
├── reports/
│   ├── Technical_Report.pdf            # 5-page publication-grade PDF report with embedded figures and tables
│   ├── technical_report.md             # Markdown technical report with comprehensive findings
│   └── generated_report.md             # Automated report generated on each pipeline execution
├── results/
│   ├── class_distribution.png          # Bar chart showing target class distribution
│   ├── suspicious_vs_legitimate.png    # Mean threat indicator values across classes
│   ├── feature_distributions.png       # Histograms of selected threat indicator distributions
│   ├── correlation_heatmap.png         # Heatmap of inter-indicator security correlations
│   ├── top_threat_indicators.png       # Top 10 indicators correlated with phishing maliciousness
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_mlp_neural_network.png
│   ├── roc_curves.png                  # Combined ROC curves with AUC metrics for all 3 models
│   ├── model_comparison_chart.png      # Multi-metric comparative performance chart
│   ├── threshold_analysis.png          # Dual-axis operational error trade-off curve (0.10 to 0.90)
│   ├── model_comparison.csv            # Tabulated quantitative benchmark metrics
│   └── metrics.json                    # Full structured results, classifications, and parameters
└── src/
    ├── main.py                         # End-to-end analysis, modeling, and automated reporting pipeline
    ├── cybersecurity_threat_detection.ipynb # Fully executed Jupyter notebook with outputs & visualizations
    ├── generate_pdf_report.py          # Programmatic ReportLab script generating Technical_Report.pdf
    └── build_notebook.py               # Automated generator script for the Jupyter notebook
```

---

## Quickstart & Execution

### 1. Clone & Activate Virtual Environment
```powershell
# Windows (PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```powershell
python src/main.py
```
*Executes data ingestion, generates all 10 figures in `results/`, trains all 3 models, runs threshold simulations, and generates `reports/generated_report.md`.*

### 3. Launch Interactive Jupyter Notebook
```powershell
jupyter notebook src/cybersecurity_threat_detection.ipynb
```
*Opens the fully executed interactive notebook documenting all 5 tasks cell-by-cell.*

### 4. Regenerate the Technical Report PDF
```powershell
python src/generate_pdf_report.py
```
*Builds the 5-page publication-quality PDF report at `reports/Technical_Report.pdf`.*

---

## Experimental Benchmark Results

### 1. Multi-Model Architecture Comparison (Test Partition: 2,211 Samples)

| Model Architecture | Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Supervised ML (Linear) | 0.9281 | 0.9344 | 0.9010 | 0.9174 | 0.9785 | **0.061s** |
| **Decision Tree** | Supervised ML (Rule-Based) | 0.9303 | 0.9249 | 0.9173 | 0.9211 | 0.9815 | **0.048s** |
| **MLP Neural Network** | Deep Learning (Dense 64x32) | **0.9656** | **0.9699** | **0.9520** | **0.9609** | **0.9944** | 2.442s |

- **Deep Learning Accuracy:** The MLP Neural Network achieved the highest overall detection rate (**96.56%**), capturing 95.20% of phishing attacks with only 29 false alarms on the test set.
- **Inference Speed:** Logistic Regression trained in 0.061s (~40x faster) and operates with sub-millisecond per-packet inference latency, ideal for inline edge firewall inspection.
- **Interpretability:** The Decision Tree provides deterministic, audit-ready decision paths that human SOC analysts can directly inspect and verify.

---

### 2. Operational Threshold & Error Trade-off Analysis (Task 5)

| Decision Threshold | Accuracy | Precision | Recall | F1-Score | False Positives (Legitimate Blocked) | False Negatives (Phishing Missed) | Operational Mode |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0.10** | 0.8616 | 0.7696 | **0.9816** | 0.8628 | 288 | 18 | Paranoid / Maximum Security |
| **0.30** | 0.9254 | 0.8983 | 0.9378 | 0.9176 | 104 | 61 | High Sensitivity |
| **0.50** | **0.9281** | **0.9344** | **0.9010** | **0.9174** | **62** | **97** | **Balanced Default** |
| **0.70** | 0.9199 | 0.9557 | 0.8592 | 0.9049 | 39 | 138 | High Availability |
| **0.90** | 0.8801 | **0.9825** | 0.7429 | 0.8460 | 13 | 252 | Lenient / Low Disruption |

- **False Positives vs False Negatives:** Lowering the threshold to 0.10 catches 98.16% of phishing threats, but generates 288 false alarms that block legitimate employees and partners. Raising the threshold to 0.90 nearly eliminates false alarms (only 13), but allows 252 phishing attacks (25.7% compromise rate) through defenses.
- **Enterprise Recommendation:** A threshold between **0.40 and 0.50** offers optimal risk-adjusted performance.

---

## Assignment Rubric Alignment (Total: 100 Marks)

| Evaluation Rubric Criteria | Max Marks | Implementation Highlights in Repository |
|:---|:---:|:---|
| **Repository Setup & Organization** | 10 | Clean root structure, virtual environment scripts, verified `requirements.txt`, git initialized, `.gitignore`, and well-organized `src/`, `data/`, `results/`, `reports/`. |
| **Threat Analysis & Visualization** | 20 | Meaningful EDA in `src/main.py` and `cybersecurity_threat_detection.ipynb`: class balance, mean indicator analysis, correlation heatmap, feature distributions, and top indicator ranking. |
| **ML Implementation** | 25 | Supervised ML baselines (**Logistic Regression** and **Decision Tree**) with complete preprocessing pipelines, confusion matrices, classification reports, and ROC-AUC metrics. |
| **DL Prototype & Comparison** | 20 | Multi-Layer Perceptron (**MLPClassifier**) deep learning prototype with comparative analysis against ML baselines covering training latency, accuracy, and practical feasibility in SOC environments. |
| **Ethical Analysis & Reflection** | 15 | Empirical threshold simulations ($0.10 \le t \le 0.90$), false-positive trade-off curve, proxy demographic/domain age bias analysis, GDPR/DPDP privacy reviews, and explainability limitations. |
| **Code Quality & Documentation** | 10 | Clean, fully documented, modular Python code (`src/main.py`), interactive Jupyter notebook, comprehensive `README.md`, and 5-page publication-grade PDF technical report (`reports/Technical_Report.pdf`). |

---

## Dataset Citation
> Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C51W2X](https://doi.org/10.24432/C51W2X). Available under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Academic Integrity Statement
This repository represents original work developed for Lab Assignment 01 of the **AI in Cyber Security Lab (ENSP355)** at **K.R. Mangalam University**. All data sources, theoretical concepts, and third-party libraries have been properly cited.
