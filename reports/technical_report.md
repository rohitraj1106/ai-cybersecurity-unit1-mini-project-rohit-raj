# Technical Report: AI-Driven Cyber Threat Awareness and Detection Prototype

**Course:** AI in Cyber Security Lab (Course Code: ENSP355)  
**Lab Assignment:** 01 (from Unit - 1) — Foundations of AI in Cyber Security  
**Student Name:** Rohit Raj  
**University:** K.R. Mangalam University  
**Faculty:** Monika Khatkar  
**Date:** October 2026  
**Artifacts Generated:** `reports/Technical_Report.pdf`, `src/cybersecurity_threat_detection.ipynb`, `results/`  

---

## 1. Problem Overview & Threat Context
Modern organizational networks and digital services are subjected to an unrelenting onslaught of cyber threats, ranging from credential harvesting and brand impersonation to automated malware distribution. Static signature engines and legacy reputation lists frequently fail against adaptive adversaries utilizing domain generation algorithms (DGAs), multi-stage redirectors, and freshly registered zero-day domains. 

Artificial Intelligence (AI) and Machine Learning (ML) empower modern Security Operations Center (SOC) defense pipelines by discovering latent patterns in domain telemetry, identifying anomalies in web infrastructure, and scoring threats dynamically. This project simulates an entry-level cyber security analyst tasked with constructing an end-to-end, reproducible threat detection engine that evaluates machine learning baselines against deep learning prototypes while critically assessing false-positive operational burdens, algorithmic bias, privacy legislation, and model explainability.

---

## 2. Methodology & Architecture
The prototype pipeline is built according to production-grade ML engineering practices:

1. **Environment Bootstrap & Dependencies:** Python 3.13 virtual environment incorporating `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `ucimlrepo`, `joblib`, and `jupyter`.
2. **Dataset Sourcing & Ingestion:** The benchmark UCI Phishing Websites dataset (Dataset ID: 327) consisting of 11,055 records across 30 integer-valued website security indicators was retrieved via `ucimlrepo` and cached in `data/phishing_websites.csv`.
3. **Target Harmonization:** The raw labels (-1 for phishing, +1 for legitimate) were mapped to binary threat semantics: `1 = Phishing (Malicious Threat)` and `0 = Legitimate (Normal Traffic)`.
4. **Data Cleaning & Preprocessing:** 
   - Missing indicators are imputed with median strategies (`SimpleImputer`).
   - Categorical/discrete features are normalized via z-score scaling (`StandardScaler`) encapsulated within Scikit-learn `Pipeline` constructs to prevent data leakage between folds.
5. **Stratified Partitioning:** An 80/20 train/test split (8,844 training samples, 2,211 testing samples) with a fixed random seed (`random_state=42`) guarantees reproducibility.
6. **Multi-Architecture Modeling:**
   - **Logistic Regression (Linear ML Baseline):** Standardized L2-regularized linear classification.
   - **Decision Tree (Rule-Based ML Classifier):** Tree-based partitioning with `max_depth=6` for human-interpretable audit rules.
   - **Multi-Layer Perceptron - MLP (Deep Learning Prototype):** Dense neural network featuring architecture (64, 32), ReLU activations, Adam optimization, and early stopping.
7. **Ethical & Operational Simulation:** Granular threshold scanning across $[0.10, 0.90]$ measuring False Positives (legitimate sites blocked) vs False Negatives (attacks missed), proxy-group bias analysis, and GDPR/DPDP privacy reviews.

---

## 3. Cyber Security Threat Landscape Analysis (EDA)
Exploratory Data Analysis revealed distinctive structural and behavioral divergences between legitimate and phishing web services:

- **Class Distribution:** The dataset contains 6,157 legitimate websites (55.7%) and 4,898 phishing websites (44.3%).
- **Divergence in Key Threat Indicators:**
  - `sslfinal_state`: Indicates SSL/TLS certificate validity, issuer trust, and certificate longevity. Legitimate websites have predominantly positive scores (+1), whereas phishing sites show negative scores (-1) due to self-signed, invalid, or absent certificates.
  - `url_of_anchor`: Percentage of hyperlink anchors (`<a>`) pointing to external or mismatched domains. Phishing sites heavily exploit cross-domain redirects to disguise destination URLs.
  - `prefix_suffix`: Presence of hyphens disguising brand names (e.g. `paypal-verification.com`). Phishing domains frequently utilize prefixed/suffixed tokens.
  - `web_traffic`: Phishing websites are ephemeral and exhibit negligible Alexa rank traffic, contrasting sharply with high-volume legitimate enterprise portals.

### Visual EDA Artifacts:
- **Class Distribution:** `results/class_distribution.png`
- **Mean Threat Indicators Comparison:** `results/suspicious_vs_legitimate.png`
- **Selected Feature Distributions:** `results/feature_distributions.png`
- **Correlation Heatmap:** `results/correlation_heatmap.png`
- **Top Threat Indicators Ranking:** `results/top_threat_indicators.png`

---

## 4. Machine Learning & Deep Learning Comparative Results

### Quantitative Benchmark (Test Partition: 2,211 instances)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.9281 | 0.9344 | 0.9010 | 0.9174 | 0.9785 | 0.061s |
| **Decision Tree** | 0.9303 | 0.9249 | 0.9173 | 0.9211 | 0.9815 | 0.048s |
| **MLP Neural Network** | **0.9656** | **0.9699** | **0.9520** | **0.9609** | **0.9944** | 2.442s |

### Performance Insights:
1. **Detection Superiority:** The **MLP Neural Network** achieved the best overall performance with **96.56% Accuracy**, **0.9609 F1-Score**, and **0.9944 ROC-AUC**, correctly identifying 933 out of 980 phishing attacks with only 29 false alarms.
2. **Computational Speed:** Logistic Regression trained in **0.061 seconds** (~40x faster than MLP) with negligible memory footprint, making it ideal for microsecond edge proxy filtering.
3. **Rule Interpretability:** The Decision Tree achieved **93.03% Accuracy** while providing transparent if-then decision boundaries that junior SOC analysts can validate in real-time.

### Performance Visualizations:
- **Receiver Operating Characteristic (ROC) Comparison:** `results/roc_curves.png`
- **Model Comparison Bar Chart:** `results/model_comparison_chart.png`
- **Confusion Matrix (Logistic Regression):** `results/confusion_matrix_logistic_regression.png`
- **Confusion Matrix (Decision Tree):** `results/confusion_matrix_decision_tree.png`
- **Confusion Matrix (MLP Neural Network):** `results/confusion_matrix_mlp_neural_network.png`

---

## 5. Ethical Reflection, Bias & Operational AI Challenges

### 5.1 The Operational Trade-Off: False Positives vs False Negatives

| Decision Threshold | Accuracy | Precision | Recall | F1-Score | False Positives (Legitimate Blocked) | False Negatives (Phishing Missed) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.10** | 0.8616 | 0.7696 | 0.9816 | 0.8628 | 288 | 18 |
| **0.20** | 0.9005 | 0.8461 | 0.9480 | 0.8941 | 169 | 51 |
| **0.30** | 0.9254 | 0.8983 | 0.9378 | 0.9176 | 104 | 61 |
| **0.40** | 0.9308 | 0.9224 | 0.9214 | 0.9219 | 76 | 77 |
| **0.50** | 0.9281 | 0.9344 | 0.9010 | 0.9174 | 62 | 97 |
| **0.60** | 0.9272 | 0.9475 | 0.8847 | 0.9150 | 48 | 113 |
| **0.70** | 0.9199 | 0.9557 | 0.8592 | 0.9049 | 39 | 138 |
| **0.80** | 0.9104 | 0.9700 | 0.8235 | 0.8907 | 25 | 173 |
| **0.90** | 0.8801 | 0.9825 | 0.7429 | 0.8460 | 13 | 252 |

- **High Security (Threshold = 0.10):** Catches 98.16% of phishing threats, but blocks 288 legitimate websites. In an enterprise, this floods the Helpdesk with unblocking requests and disrupts normal business operations.
- **High Permissiveness (Threshold = 0.90):** Reduces false positives to only 13 instances, but allows 252 phishing attacks to bypass defenses (25.7% compromise rate), exposing the organization to credential harvesting and data exfiltration.
- **Recommended Enterprise Calibration:** Calibrating the threshold at **0.40–0.50** maximizes F1-score (~0.92) while balancing business continuity and threat mitigation.
- **Visualization:** `results/threshold_analysis.png`

### 5.2 Algorithmic Bias on Domain Age & Proxies
Stratified analysis revealed that newly registered domains (`age_of_domain = -1`) are heavily penalized by the classifiers because attackers predominantly use freshly minted domains. However, legitimate new business startups and newly launched academic portals also share this exact characteristic. Automated blocking without contextual review unfairly suppresses new enterprises. Systems must incorporate reputation aging curves and manual whitelisting to mitigate algorithmic bias.

### 5.3 Privacy Concerns & Legal Compliance
Deploying live URL inspection requires monitoring HTTP headers, DNS request logs, and employee browsing histories. Under the **EU General Data Protection Regulation (GDPR)** and India's **Digital Personal Data Protection (DPDP) Act 2023**, full browsing URLs often contain personal identifiers, session tokens, and health/financial metadata. Defenses must enforce:
- Client-side cryptographic hashing of URL query parameters.
- Data minimization: extracting structural indicators rather than storing raw URLs.
- Role-based access control (RBAC) and automated retention purges (e.g. 30-day log rotation).

### 5.4 Explainability & SOC Triage Feasibility
While the MLP neural network delivers peak accuracy, its complex non-linear weights prevent tier-1 analysts from understanding the root cause of an alert. We propose a **hybrid operational pipeline**:
- Tier 1: Logistic Regression and Decision Tree models inspect traffic in real-time, instantly blocking clear malicious domains and providing audit explanations.
- Tier 2: Ambiguous or borderline alerts (probabilities between 0.35 and 0.65) are routed to the MLP Neural Network and placed in an analyst queue with a mandatory **Human-in-the-Loop (HITL)** appeal path.

---

## 6. Project Deliverables Verification

- **Source Code:** `src/main.py` (End-to-end executable pipeline)
- **Interactive Notebook:** `src/cybersecurity_threat_detection.ipynb` (Fully executed notebook with embedded visualizations)
- **PDF Report:** `reports/Technical_Report.pdf` (Publication-grade 5-page PDF with embedded charts and metrics)
- **Visual Artifacts:** `results/*.png` (10 high-resolution analytical figures)
- **Environment & Setup:** `requirements.txt`, `.gitignore`, `README.md`

---

## 7. References
1. Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. NIST Special Publication 800-61 Rev. 2 (2012). *Computer Security Incident Handling Guide*. National Institute of Standards and Technology.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
