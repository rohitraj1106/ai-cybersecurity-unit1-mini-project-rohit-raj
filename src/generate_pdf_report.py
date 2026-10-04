"""
Automated professional PDF report generator for:
Lab Assignment 01: Foundations of AI in Cyber Security (ENSP355)
K.R. Mangalam University
"""
import os
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
PDF_PATH = REPORTS_DIR / "Technical_Report.pdf"

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "K.R. Mangalam University | AI in Cyber Security Lab (ENSP355) — Technical Report")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 35, footer_text)
        self.drawString(54, 35, "Confidential — Academic Lab Assessment — Student: Rohit Raj")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 47, 558, 47)
        self.restoreState()

def build_pdf():
    # Load metrics
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0d3b66'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#f95738'),
        spaceAfter=14
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0d3b66'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2b5c8f'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#222222'),
        leftIndent=15,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#555555'),
        alignment=1, # Centered
        spaceAfter=8
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#222222')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0d3b66')
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # Title Banner
    story.append(Paragraph("End-to-End Mini Project: Building an AI-Driven Cyber Threat Awareness and Detection Prototype", title_style))
    story.append(Paragraph("Lab Assignment : 01 (from Unit - 1) — Foundations of AI in Cyber Security", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d3b66"), spaceAfter=10))

    # Meta Table
    meta_data = [
        [
            Paragraph("<b>Course:</b> AI in Cyber Security Lab", table_cell_style),
            Paragraph("<b>Student Name:</b> Rohit Raj", table_cell_style)
        ],
        [
            Paragraph("<b>Course Code:</b> ENSP355", table_cell_style),
            Paragraph("<b>University:</b> K.R. Mangalam University", table_cell_style)
        ],
        [
            Paragraph("<b>Weightage:</b> 20% of Lab Assessment", table_cell_style),
            Paragraph("<b>Faculty:</b> Monika Khatkar", table_cell_style)
        ],
        [
            Paragraph("<b>Repository:</b> ai-cybersecurity-unit1-mini-project", table_cell_style),
            Paragraph("<b>Date:</b> October 2026", table_cell_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f4f7f6")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e4e8")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Problem Overview
    story.append(Paragraph("1. Problem Overview & Real-World Cyber Threat Context", h1_style))
    story.append(Paragraph(
        "Modern corporate and governmental infrastructures face escalating threat volumes from sophisticated social engineering, "
        "credential-harvesting websites, and automated evasive phishing attacks. Traditional defense perimeter mechanisms (such as static "
        "reputation blocklists and signature-based heuristic rules) fail against dynamic domain generation algorithms (DGAs), multi-stage "
        "redirectors, and novel zero-hour malicious domains. This mini-project simulates an entry-level cyber security analyst developing "
        "an intelligent, data-driven security analysis and threat classification engine capable of distinguishing legitimate web resources "
        "from cyber threats while critically assessing operational false positives, bias, and deep learning feasibility.",
        body_style
    ))

    # 2. Methodology & Architecture
    story.append(Paragraph("2. System Architecture & Methodology", h1_style))
    story.append(Paragraph(
        "The prototype pipeline was designed using a rigorous, reproducible machine learning engineering workflow:",
        body_style
    ))
    story.append(Paragraph("• <b>Environment Bootstrap:</b> Python 3.13 virtual environment with Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, UCIMLRepo, and Jupyter.", bullet_style))
    story.append(Paragraph("• <b>Dataset Sourcing & Harmonization:</b> Ingestion of the verified UCI Phishing Websites benchmark dataset (ID 327, 11,055 instances, 30 features). Class labels were standardized to <code>1 = Phishing (Malicious Threat)</code> and <code>0 = Legitimate (Normal Traffic)</code>.", bullet_style))
    story.append(Paragraph("• <b>Data Cleansing & Preprocessing:</b> Median imputation for missing numeric indicators, one-hot encoding for categorical variables, and z-score standard normalization (StandardScaler) inside encapsulated Scikit-learn Pipelines to avoid data leakage.", bullet_style))
    story.append(Paragraph("• <b>Stratified Data Partitioning:</b> 80% training (8,844 samples) and 20% testing (2,211 samples) using a fixed random seed (42) to guarantee reproducible comparative baselines.", bullet_style))
    story.append(Paragraph("• <b>Multi-Paradigm Modeling:</b> Training and evaluating a linear baseline (Logistic Regression), a non-linear rule tree (Decision Tree), and a deep multi-layer neural network (MLPClassifier).", bullet_style))
    story.append(Paragraph("• <b>Ethical & Operational Simulation:</b> Detailed threshold variation (0.10 to 0.90) measuring the real-world operational trade-off between False Positives (user lockouts) and False Negatives (breaches), alongside proxy-group fairness and privacy analysis.", bullet_style))

    story.append(Spacer(1, 8))

    # 3. Threat Landscape Analysis (EDA)
    story.append(Paragraph("3. Cyber Security Threat Landscape Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "Exploratory Data Analysis revealed key behavioral anomalies that distinguish phishing campaigns from benign web infrastructure. "
        "The dataset exhibits a balanced distribution with 6,157 legitimate websites (55.7%) and 4,898 phishing websites (44.3%).",
        body_style
    ))

    # Embed EDA images side-by-side or stacked
    img_class = str(RESULTS_DIR / "class_distribution.png")
    img_means = str(RESULTS_DIR / "suspicious_vs_legitimate.png")
    if os.path.exists(img_class) and os.path.exists(img_means):
        eda_table = Table([
            [Image(img_class, width=240, height=155), Image(img_means, width=255, height=155)],
            [Paragraph("Figure 1: Target Class Distribution", caption_style), Paragraph("Figure 2: Mean Indicator Values by Website Class", caption_style)]
        ], colWidths=[250, 254])
        eda_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(eda_table)

    story.append(Paragraph(
        "<b>Key Threat Indicator Insights:</b> As shown in Figure 2, features such as <code>sslfinal_state</code> (validity of SSL/TLS certificate, issuer trust, and certificate age), "
        "<code>url_of_anchor</code> (percentage of hyperlinks pointing to differing external domains), <code>prefix_suffix</code> (presence of hyphens disguising brand names), and "
        "<code>web_traffic</code> (Alexa global ranking) exhibit sharp divergence between classes. Legitimate domains show predominantly positive values (+1), whereas phishing "
        "domains show negative encoded values (-1), confirming strong discriminative signal.",
        body_style
    ))

    # Heatmap & Top indicators
    img_corr = str(RESULTS_DIR / "correlation_heatmap.png")
    img_top = str(RESULTS_DIR / "top_threat_indicators.png")
    if os.path.exists(img_corr) and os.path.exists(img_top):
        story.append(Spacer(1, 4))
        eda_table2 = Table([
            [Image(img_corr, width=250, height=180), Image(img_top, width=245, height=180)],
            [Paragraph("Figure 3: Indicator Correlation Heatmap", caption_style), Paragraph("Figure 4: Top Features Correlated with Phishing Target", caption_style)]
        ], colWidths=[252, 252])
        eda_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(eda_table2)

    story.append(PageBreak())

    # 4. Machine Learning & Deep Learning Implementation & Comparison
    story.append(Paragraph("4. Machine Learning & Deep Learning Model Implementation", h1_style))
    story.append(Paragraph(
        "To rigorously address Tasks 3 and 4, three distinct architectural paradigms were trained and validated on the identical 80/20 stratified split:",
        body_style
    ))
    story.append(Paragraph("1. <b>Logistic Regression (Supervised ML Baseline):</b> Scaled linear model with L2 regularization, serving as an ultra-low-latency classification benchmark.", bullet_style))
    story.append(Paragraph("2. <b>Decision Tree Classifier (Rule-Based ML):</b> Non-linear partitioning with maximum depth 6, generating auditable if-then decision trees suitable for tier-1 security analyst inspection.", bullet_style))
    story.append(Paragraph("3. <b>Multi-Layer Perceptron - MLP (Deep Learning Prototype):</b> Neural network featuring hidden architecture (64, 32), ReLU activation functions, Adam optimizer, and early stopping to prevent overfitting.", bullet_style))

    story.append(Paragraph("Quantitative Comparative Evaluation", h2_style))
    story.append(Paragraph(
        "Each architecture was evaluated across test Accuracy, Precision (minimizing false alarms), Recall (catching threats), F1-Score, ROC-AUC, and Training Execution Latency:",
        body_style
    ))

    # Comparison Table
    table_headers = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Train Time"]
    table_rows = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in table_headers]]
    
    for m in metrics["models"]:
        table_rows.append([
            Paragraph(f"<b>{m['model']}</b>", table_cell_bold),
            Paragraph(f"{m['accuracy']:.4f}", table_cell_style),
            Paragraph(f"{m['precision']:.4f}", table_cell_style),
            Paragraph(f"{m['recall']:.4f}", table_cell_style),
            Paragraph(f"{m['f1_score']:.4f}", table_cell_style),
            Paragraph(f"{m['roc_auc']:.4f}", table_cell_style),
            Paragraph(f"{m['training_time_seconds']:.3f}s", table_cell_style)
        ])
    
    comp_table = Table(table_rows, colWidths=[130, 60, 60, 60, 60, 64, 70])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d3b66")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(comp_table)
    story.append(Paragraph("Table 1: Performance Benchmark on Stratified Test Set (2,211 instances)", caption_style))

    # Embed ROC Curve and Model Comparison Chart
    img_roc = str(RESULTS_DIR / "roc_curves.png")
    img_comp = str(RESULTS_DIR / "model_comparison_chart.png")
    if os.path.exists(img_roc) and os.path.exists(img_comp):
        story.append(Spacer(1, 4))
        eval_table = Table([
            [Image(img_roc, width=250, height=170), Image(img_comp, width=250, height=170)],
            [Paragraph("Figure 5: Receiver Operating Characteristic (ROC) Curves", caption_style), Paragraph("Figure 6: Multi-Metric Performance by Architecture", caption_style)]
        ], colWidths=[252, 252])
        eval_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(eval_table)

    # Confusion matrices
    img_cm_lr = str(RESULTS_DIR / "confusion_matrix_logistic_regression.png")
    img_cm_dt = str(RESULTS_DIR / "confusion_matrix_decision_tree.png")
    img_cm_mlp = str(RESULTS_DIR / "confusion_matrix_mlp_neural_network.png")
    if os.path.exists(img_cm_lr) and os.path.exists(img_cm_dt) and os.path.exists(img_cm_mlp):
        cm_table = Table([
            [Image(img_cm_lr, width=165, height=135), Image(img_cm_dt, width=165, height=135), Image(img_cm_mlp, width=165, height=135)],
            [Paragraph("Figure 7a: CM — Logistic Regression", caption_style),
             Paragraph("Figure 7b: CM — Decision Tree", caption_style),
             Paragraph("Figure 7c: CM — MLP Neural Network", caption_style)]
        ], colWidths=[168, 168, 168])
        cm_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(cm_table)

    story.append(PageBreak())

    # 5. Ethical Reflection & AI Challenges
    story.append(Paragraph("5. Ethical Reflection, Bias & Operational AI Challenges", h1_style))
    story.append(Paragraph(
        "Deploying artificial intelligence inside automated cyber defense environments introduces critical ethical, legal, and operational dilemmas. "
        "Task 5 mandates evaluating the direct consequences of classification thresholds, algorithmic bias, privacy violations, and explainability deficits.",
        body_style
    ))

    story.append(Paragraph("5.1 Operational Error Simulation: False Positives vs False Negatives", h2_style))
    story.append(Paragraph(
        "In security operations, errors carry asymmetric real-world consequences:",
        body_style
    ))
    story.append(Paragraph("• <b>False Positives (Legitimate Users Blocked):</b> A legitimate business domain, customer checkout portal, or partner API is incorrectly labeled as malicious. This causes immediate operational blockage, direct revenue loss, brand reputation damage, and severe Helpdesk alert fatigue.", bullet_style))
    story.append(Paragraph("• <b>False Negatives (Phishing Attacks Missed):</b> An active phishing site bypasses detection, enabling credential theft, multi-factor authentication (MFA) token hijacking, ransomware distribution, and catastrophic organizational breach.", bullet_style))

    # Threshold Table
    thresh_headers = ["Threshold", "Accuracy", "Precision", "Recall", "F1-Score", "False Positives (Blocked)", "False Negatives (Missed)"]
    thresh_rows = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in thresh_headers]]
    for t in metrics["threshold_analysis"]:
        thresh_rows.append([
            Paragraph(f"{t['threshold']:.2f}", table_cell_style),
            Paragraph(f"{t['accuracy']:.4f}", table_cell_style),
            Paragraph(f"{t['precision']:.4f}", table_cell_style),
            Paragraph(f"{t['recall']:.4f}", table_cell_style),
            Paragraph(f"{t['f1_score']:.4f}", table_cell_style),
            Paragraph(f"{t['false_positives_legitimate_flagged']}", table_cell_bold),
            Paragraph(f"{t['false_negatives_phishing_missed']}", table_cell_bold),
        ])
    thresh_table = Table(thresh_rows, colWidths=[55, 65, 65, 65, 65, 95, 94])
    thresh_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f95738")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7de")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fdf7f5")]),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(thresh_table)
    story.append(Paragraph("Table 2: Threshold Variation & Error Trade-off Analysis (Logistic Regression)", caption_style))

    # Embed Threshold Analysis Chart
    img_thresh = str(RESULTS_DIR / "threshold_analysis.png")
    if os.path.exists(img_thresh):
        story.append(Image(img_thresh, width=500, height=190))
        story.append(Paragraph("Figure 8: Operational Error Count and Metric Curves across Decision Thresholds (0.10 to 0.90)", caption_style))

    story.append(Paragraph("5.2 Algorithmic Bias and Representativeness Limitations", h2_style))
    story.append(Paragraph(
        "Our proxy-group evaluation (analyzing performance stratified by domain age and IP-address presence) highlights that models "
        "trained on historical data can exhibit systematic bias against legitimate new businesses. For instance, newly incorporated startup "
        "domains inherently have an <code>age_of_domain = -1</code> (less than 6 months old), which strongly correlates with malicious kits in "
        "training data. Consequently, legitimate startups face higher baseline false-positive rates, risking anti-competitive suppression. "
        "Models must be retrained continuously with diverse, temporally balanced datasets to avoid static bias.",
        body_style
    ))

    story.append(Paragraph("5.3 Privacy Concerns & Regulatory Compliance (GDPR / DPDP Act)", h2_style))
    story.append(Paragraph(
        "Real-world deployment of URL inspection systems requires intercepting user DNS queries, HTTP GET headers, and full browsing histories. "
        "Such deep telemetry logging creates substantial privacy hazards under privacy frameworks (such as the EU GDPR and India's Digital Personal Data Protection Act 2023). "
        "Organizations must enforce strict data minimization, client-side cryptographic hashing of query parameters, strict role-based access control (RBAC), and automated log retention purges.",
        body_style
    ))

    story.append(Paragraph("5.4 Explainability & SOC Operational Feasibility", h2_style))
    story.append(Paragraph(
        "While the <b>MLP Neural Network</b> achieved the peak detection rate (96.56% accuracy), it functions as an opaque black box. "
        "In a live Security Operations Center (SOC), tier-1 analysts cannot afford to blindly accept model alerts; they require actionable explanations "
        "(e.g., 'flagged due to anchor URLs pointing to unverified IP address'). For this reason, a hybrid architecture is recommended: "
        "deploying <b>Logistic Regression</b> or <b>Decision Trees</b> at the edge for millisecond triage and human auditing, while routing ambiguous "
        "or high-confidence borderline alerts to the <b>MLP Neural Network</b> with a mandatory Human-in-the-Loop review path.",
        body_style
    ))

    # 6. Conclusion & References
    story.append(Paragraph("6. Conclusion & References", h1_style))
    story.append(Paragraph(
        "This project successfully developed, evaluated, and documented an end-to-end AI-driven threat detection prototype. "
        "Key conclusions include: (1) Machine learning achieves high accuracy (>92%) in distinguishing phishing attacks from benign web traffic; "
        "(2) Neural architectures improve detection performance (~96.5%) at the cost of opacity and compute; (3) Threshold calibration is essential "
        "to prevent catastrophic user lockouts; and (4) Ethical security governance requires human oversight and privacy protections.",
        body_style
    ))
    story.append(Paragraph("<b>Academic References:</b>", h2_style))
    story.append(Paragraph("1. Mohammad, R. & McCluskey, L. (2012). <i>Phishing Websites</i> [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X", bullet_style))
    story.append(Paragraph("2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.", bullet_style))
    story.append(Paragraph("3. NIST Special Publication 800-61 Rev. 2 (2012). <i>Computer Security Incident Handling Guide</i>. National Institute of Standards and Technology.", bullet_style))
    story.append(Paragraph("4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). <i>Deep Learning</i>. MIT Press.", bullet_style))

    # Build PDF with dynamic page numbering
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technical Report PDF successfully generated at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
