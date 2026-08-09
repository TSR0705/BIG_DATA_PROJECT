# Loan Default Prediction Using Big Data Analytics
### Technical Feasibility Report, Short PRD & Technical Survey
**Final-Year Big Data + AI/ML Project**

---

## 1. Executive Summary

This report evaluates the technical feasibility of building an **end-to-end, explainable loan-default risk prediction system** that combines Big Data engineering (Apache Spark), machine learning (gradient-boosted trees), explainable AI (SHAP), and a decision-support dashboard.

**Verdict in one line:** The project is **highly feasible** for a pre final-year student on a single laptop (16 GB RAM) using the **Home Credit Default Risk** dataset (~307,511 applicants, 8 relational tables, ~2.7 GB) processed with **PySpark**, modeled with **XGBoost/LightGBM**, explained with **SHAP**, and served through **FastAPI + Streamlit + PostgreSQL**, containerized with **Docker**. Kafka, Hadoop/HDFS, and cloud clusters are **not required** for the MVP and are positioned as optional advanced extensions, not core dependencies.

---

## 2. Project Definition (PRD)

### 2.1 Problem
Manual and legacy statistical loan approval processes are slow, inconsistent, and often provide no explanation for a rejection, creating regulatory and customer-trust risk. Financial institutions need systems that assess default risk accurately, at scale, across heterogeneous, multi-table historical data, while remaining explainable to loan officers and auditors.

### 2.2 Proposed Solution
A pipeline that ingests raw application, bureau, and repayment-history data; cleans and joins it at scale with Spark; engineers domain features (DTI, LTI, delinquency counts, credit utilization); trains a gradient-boosted classifier; explains every prediction with SHAP; and exposes the result through an API and dashboard for a loan officer to review as a **decision-support** tool (not an autonomous approver).

### 2.3 Target Users
Loan officers / underwriters, risk analysts, and (secondarily) academic evaluators assessing the project's technical depth.

### 2.4 Main Objective
Predict the probability that a loan applicant will default, and explain *why*, using a pipeline that is architecturally scalable even though the demo dataset fits on a laptop.

### 2.5 Expected Output
A working system: cleaned feature store → trained model (with saved artifact) → REST API (`/predict`, `/explain`) → interactive dashboard showing risk score, SHAP explanation, portfolio trends, and model performance.

### 2.6 Why This Is a Big Data Project, Not "Just ML"
A single flat CSV run through `pandas.read_csv()` and `scikit-learn` would be a pure ML project. This project qualifies as Big Data + Data Engineering because:
- The source data is **relational and multi-table** (up to 8 linked tables, one row per historical transaction/month, tens of millions of rows in the bureau-balance and installment tables) — this requires distributed joins and aggregation, not a single in-memory table.
- **Feature engineering at scale**: aggregating transaction-level history (min/max/mean/sum over previous loans) per customer is a classic Spark groupBy/aggregate workload that scales poorly in pure pandas as row counts grow.
- The **architecture is designed to scale horizontally** (partitioned Parquet, Spark jobs) even though the specific competition dataset is laptop-sized — this is explicitly addressed in Section 3 so the claim is not overstated.
- Optional streaming extension (Kafka + Spark Structured Streaming) demonstrates real-time Big Data ingestion patterns.

The system therefore genuinely exercises **Big Data engineering + Machine Learning + Explainable AI**, not ML alone.

---

## 3. Technical Feasibility

### 3.1 Hardware Requirements

| Resource | MVP (local) | Recommended |
|---|---|---|
| RAM | 8 GB (tight) | 16 GB |
| CPU | 4 cores | 4–8 cores |
| Disk | 20 GB free | 40 GB (raw + Parquet + models) |
| GPU | Not required | Not required (XGBoost/LightGBM CPU is sufficient at this scale) |

### 3.2 Software Requirements
Python 3.10+, PySpark 3.5.x, Java 11/17 (Spark dependency), XGBoost, LightGBM (optional), SHAP, FastAPI, Streamlit, PostgreSQL 15+, Docker, Git.

### 3.3 Dataset Availability
All three candidate datasets (Home Credit, Lending Club, Give Me Some Credit) are freely downloadable from Kaggle with a free account — no procurement risk (see Section 4).

### 3.4 Computational & Storage Requirements
Home Credit's full relational dataset is **~2.7 GB compressed / ~1 GB per largest single table** (`bureau_balance.csv` has ~27 million rows). This is comfortably handled by:
- Spark in **local mode** (`local[*]`) on a single laptop — Spark's partitioned, lazy-evaluation engine handles this without a cluster.
- Alternatively, `pandas` with chunked reads for the smaller tables, if Spark setup proves an obstacle — see risk mitigation below.

### 3.5 Is a Local Machine Sufficient? Is Cloud Necessary?
**A local machine is sufficient for the entire MVP.** Spark's local mode simulates a distributed cluster on a single machine's cores and is standard practice for Big Data coursework and prototyping. Cloud deployment (a small VM, or free tiers of AWS/GCP/Azure, or Databricks Community Edition) is **optional** — useful for demonstrating "real" distributed deployment to evaluators, but not required to complete or defend the project.

### 3.6 Development Complexity & Estimated Difficulty
- Data engineering (Spark joins/aggregations across 8 tables): **Medium-High** — the main technical challenge.
- ML modeling (XGBoost on tabular data): **Low-Medium** — well-documented, abundant reference notebooks exist.
- SHAP explainability: **Low** — mature library, near plug-and-play with tree models.
- API + dashboard: **Low-Medium** — standard FastAPI/Streamlit patterns.
- **Overall: Medium**, appropriate for a final-year capstone.

### 3.7 Technical Risks & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Spark/Java environment setup fails on student laptop | Blocks pipeline | Provide Docker Compose with pre-built Spark image; fallback to `pandas` + `pyarrow` for MVP if Spark install is infeasible, clearly noting the trade-off |
| 27M-row `bureau_balance` table causes memory pressure | Slow iteration | Aggregate early (per `SK_ID_BUREAU`) before joining to main table; use Parquet + partitioning, not repeated CSV reads |
| Severe class imbalance (~8% defaults) causes misleading accuracy | Poor model trusted incorrectly | Use ROC-AUC/PR-AUC/F1 as primary metrics, not accuracy; apply class weighting; see Section 10 |
| SHAP computation slow on full dataset | Delays dashboard | Use `TreeExplainer` (fast, exact for tree models) and sample for global plots |
| Scope creep into Kafka/Kubernetes/cloud clusters | Missed deadline | Explicit MVP/Advanced split (Section 21); advanced items are optional stretch goals only |

### 3.8 MUST-HAVE / SHOULD-HAVE / OPTIONAL Components

| Tier | Components |
|---|---|
| **MUST-HAVE** | Dataset (Home Credit), PySpark for ETL/joins, feature engineering, XGBoost model, evaluation with AUC/F1/PR-AUC, SHAP explanations, FastAPI serving layer, Streamlit dashboard, Docker for reproducibility, Git |
| **SHOULD-HAVE** | PostgreSQL to persist predictions, LightGBM as comparison model, class-imbalance handling (SMOTE/class-weights), CI-friendly project structure |
| **OPTIONAL/ADVANCED** | Kafka + Spark Structured Streaming, cloud VM deployment, automated retraining, drift detection, fairness auditing dashboard, Kubernetes |

---

## 4. Dataset Survey and Selection

### 4.1 Candidate Datasets

**A. Home Credit Default Risk** (Kaggle competition, Home Credit Group)
- URL: https://www.kaggle.com/c/home-credit-default-risk/data
- Records: <cite index="2-1">307,511 customers in the main application table</cite>, with <cite index="2-1">92% non-default and roughly 8% default class</cite>.
- Tables/features: <cite index="5-1">8 relational tables in total</cite>, with <cite index="5-1">221 raw features when all tables are combined</cite>; the main table alone has 122 columns. The largest supporting table, credit bureau monthly balances, has roughly 27 million rows.
- Format: CSV, multiple linked tables joined via `SK_ID_CURR`/`SK_ID_BUREAU`/`SK_ID_PREV`.
- Data type: Tabular, mixed numeric/categorical, static application data + historical behavioral data.
- Missing values: Extensive missing values across many columns (documented in the competition data dictionary).
- Class imbalance: Present (~8% positive class).
- Big Data suitability: **High** — genuinely multi-table with tens of millions of rows in aggregate, ideal for Spark joins/aggregation.
- ML suitability: High — this was a well-studied Kaggle competition with strong reference baselines.
- Licensing: Kaggle competition rules license (free for research/educational use; redistribution restrictions apply per Kaggle terms).
- Advantages: Richest feature set, multiple related tables (genuine Big Data ETL exercise), realistic missingness, strong community benchmarks.
- Limitations: Requires careful multi-table joining; steeper learning curve; anonymized/obscured feature meanings for some columns.

**B. Lending Club Loan Dataset**
- URL: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Records/size: <cite index="12-1">Approximately 890,000 loan records with about 75 attributes</cite>, spanning <cite index="9-1">2007 through recent years, split into accepted and rejected loan files</cite>.
- Format: CSV, single wide table (plus a separate rejected-loans file).
- Target variable: `loan_status` (Fully Paid / Charged Off / Default / etc. — must be binarized).
- Missing values: Present, particularly in employment and joint-application fields.
- Class imbalance: Moderate (depends on how "default" is defined from `loan_status`).
- Big Data suitability: **Medium** — large row count but effectively a single flat table, so distributed *joins* are not demonstrated; Spark is still useful for scale but the "Big Data" case is weaker than Home Credit.
- ML suitability: High — widely used benchmark, real (not obfuscated) financial features like interest rate, grade, DTI.
- Licensing: Publicly shared under Kaggle dataset terms.
- Advantages: Real interpretable feature names, large volume, good for a single-table Spark ETL + feature engineering story.
- Limitations: Single-table structure weakens the multi-table Big Data narrative; label definition (`loan_status` → default) requires a documented, defensible business rule.

**C. Give Me Some Credit**
- URL: https://www.kaggle.com/c/give-me-some-credit (2011 Kaggle competition)
- Records: <cite index="20-1">About 250,000 anonymous borrower records total, split into 150,000 labeled training rows (10,026 positive / 139,974 negative) and 101,503 unlabeled test rows</cite>, with <cite index="19-1">11 columns per row</cite> and a target of `SeriousDlqin2yrs`.
- Size: <cite index="20-1">Roughly 15 MB uncompressed, ~5 MB compressed</cite>.
- Class imbalance: <cite index="20-1">About 6.7% positive class (roughly a 1:14 ratio)</cite>.
- Big Data suitability: **Low** — small, single flat table; not suitable as the primary dataset for a Big Data project.
- ML suitability: High — clean, simple, excellent for a *baseline*/teaching example.
- Advantages: Very easy to prototype the ML/SHAP pipeline quickly.
- Limitations: Too small/simple to justify Spark or any Big Data claim; only 11 columns limits feature-engineering depth.

### 4.2 Primary Dataset Selection and Justification

**Selected primary dataset: Home Credit Default Risk.**

Justification:
1. It is the only candidate with a genuine **multi-table relational structure** (8 tables), which is what makes Spark-based ETL, joins, and aggregation a real requirement rather than a decorative add-on.
2. Its scale (hundreds of thousands of applicants, tens of millions of historical transaction rows) is large enough to make single-machine pandas workflows slow/fragile, which is the honest justification for distributed processing described below.
3. It has the richest feature space for the derived-feature and explainability work required in Sections 5–8.
4. It is a well-benchmarked competition dataset, so results can be sanity-checked against published leaderboards and papers (Section 18).

*Give Me Some Credit* is retained as a lightweight **secondary/validation dataset** for early pipeline smoke-testing (fast iteration before scaling to Home Credit), and *Lending Club* is noted as a viable **alternative primary dataset** if a simpler, single-table Spark pipeline is preferred over multi-table joins.

### 4.3 Honest Big-Data Framing
None of these datasets are petabyte-scale, and this report does not claim otherwise. The Big Data case rests on: (a) multi-table relational joins at row counts (tens of millions) that stress single-machine pandas memory; (b) partitioned, columnar (Parquet) storage and lazy-evaluated distributed transformations via Spark; (c) an architecture designed to scale horizontally by adding executors/partitions if data volume grows (e.g., a real bank's multi-year transaction history), which is demonstrated on this dataset at a reduced but representative scale.

---

## 5. Dataset & Feature Analysis

### 5.1 Feature Categories (Home Credit)

| Category | Example fields | Represents |
|---|---|---|
| Demographic | `CODE_GENDER`, `DAYS_BIRTH`, `CNT_FAM_MEMBERS`, `NAME_FAMILY_STATUS` | Applicant's personal/household profile |
| Financial | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` | Income and loan sizing |
| Employment | `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `OCCUPATION_TYPE` | Income stability |
| Credit history | `bureau.csv` fields: `CREDIT_ACTIVE`, `CREDIT_DAY_OVERDUE`, `AMT_CREDIT_SUM` | External credit bureau record |
| Loan information | `NAME_CONTRACT_TYPE`, `NAME_CASH_LOAN_PURPOSE` | Loan product details |
| Repayment/payment history | `installments_payments.csv`: `DAYS_INSTALMENT`, `AMT_PAYMENT` | Past on-time/late behavior |
| Previous credit (internal) | `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv` | Prior relationship with the same lender |
| External/behavioral | `EXT_SOURCE_1/2/3` (external normalized credit scores) | Third-party risk signals |

For each field, preprocessing follows the pipeline in Section 8: numeric fields are imputed (median) and scaled where needed for non-tree models; categoricals are one-hot or target/frequency encoded; the expected relationship with default is stated per feature during EDA and validated with SHAP after training rather than assumed a priori.

### 5.2 Proposed Derived Features

| Derived feature | Formula / logic | Contribution |
|---|---|---|
| Debt-to-Income (DTI) | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Captures repayment burden relative to earnings — one of the strongest classical credit-risk signals |
| Loan-to-Income (LTI) | `AMT_CREDIT / AMT_INCOME_TOTAL` | Captures over-leveraging risk |
| Credit utilization | `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL` (from credit_card_balance) | High utilization is a known early-warning indicator |
| Previous delinquency count | Count of `CREDIT_DAY_OVERDUE > 0` in bureau history | Direct proxy for past default behavior |
| Payment history score | Aggregated (`AMT_PAYMENT` − `AMT_INSTALMENT`) trend, % on-time payments | Summarizes repayment discipline into one feature |
| Number of active loans | Count of `CREDIT_ACTIVE == 'Active'` in bureau | Current debt load / financial stress |
| Installment burden | `AMT_ANNUITY / (AMT_INCOME_TOTAL / 12)` | Monthly cash-flow strain |
| Credit history length | `DAYS_CREDIT` span (oldest to newest bureau record) | Longer, stable history generally lowers risk |

### 5.3 Fairness/Privacy Treatment of Sensitive Attributes
`CODE_GENDER`, `DAYS_BIRTH` (age), and family/marital status are present in the raw data. Per Section 13, the recommended approach is: **exclude gender and marital status from the primary model's training features**; retain `DAYS_BIRTH` only if legally permitted for credit-risk modeling in the target jurisdiction, and in all cases **evaluate model outcomes across these groups post-hoc** (disparate impact / demographic parity checks) even when the attribute is excluded from training, since correlated proxy features (e.g., occupation, address) can reintroduce bias.

---

## 6. Proposed Architecture

```
Data Source (Home Credit CSVs)
        ↓
Data Ingestion (PySpark batch read → Parquet)
        ↓
Distributed Storage (Partitioned Parquet on local disk / optional HDFS or S3)
        ↓
Data Cleaning (PySpark: nulls, duplicates, dtype casting)
        ↓
Feature Engineering (PySpark aggregations across 8 tables → single feature table)
        ↓
ML Training (XGBoost / LightGBM, trained on the engineered Pandas/Spark-exported table)
        ↓
Model Evaluation (AUC, PR-AUC, F1, calibration)
        ↓
Explainable AI (SHAP TreeExplainer — global + local explanations)
        ↓
API (FastAPI: /predict, /explain)
        ↓
Dashboard (Streamlit: risk scoring, explanations, monitoring views)
        ↓
Deployment (Docker Compose: API + dashboard + PostgreSQL, local or single cloud VM)
```

### 6.1 Component-by-Component Justification

| Layer | Technology | Why needed | Function | Mandatory? | Simpler alternative |
|---|---|---|---|---|---|
| Ingestion/ETL | PySpark | Multi-table joins at 10M+ row scale | Read CSV → Parquet, join tables, aggregate | **Yes** | `pandas` + `pyarrow` chunking (loses the "Big Data" justification) |
| Storage | Parquet (local) | Columnar, compressed, partition-friendly | Persist cleaned/aggregated feature table | **Yes** | Plain CSV (slower, larger, no schema) |
| Distributed FS | HDFS / S3 | Only relevant at true multi-node scale | Would host Parquet across a cluster | **Optional** | Local filesystem — sufficient for this dataset size |
| Streaming ingestion | Kafka | Only needed for real-time scoring | Stream new applications for live scoring | **Optional (advanced)** | Batch API calls |
| Stream processing | Spark Structured Streaming | Pairs with Kafka | Real-time feature computation | **Optional (advanced)** | N/A |
| ML framework | XGBoost | Best accuracy/interpretability/speed trade-off on tabular credit data (see Section 7) | Train the risk classifier | **Yes** | Spark MLlib GBTClassifier (native-Spark alternative) |
| Explainability | SHAP | Regulatory expectation for credit decisions; exact for tree models | Local + global explanations | **Yes** | LIME (approximate, less stable) |
| Backend API | FastAPI | Async, typed, auto-generated docs, fast | Serve predictions/explanations | **Yes** | Flask (simpler but less performant/typed) |
| Frontend | Streamlit | Rapid Python-native dashboarding, ideal for a student project | Risk dashboard, charts | **Yes** | Dash / a custom React app (more effort, not warranted for MVP) |
| Database | PostgreSQL | Durable, relational, easy to query for reporting | Store prediction logs, model metadata | **Should-have** | SQLite (fine for a purely local demo) |
| Containerization | Docker | Reproducibility across grading environments | Package API + dashboard + DB | **Yes** | Manual `venv` setup (fragile across machines) |
| Orchestration | Kubernetes | Only relevant at production multi-service scale | Would manage container scaling | **No** | Docker Compose is sufficient |
| Version control | Git/GitHub | Standard practice, required for any evaluated capstone | Source control, collaboration | **Yes** | — |
| Cloud (AWS/Azure/GCP) | Optional VM | Demonstrates deployment beyond localhost | Host containers publicly | **Optional** | Local demo during defense is acceptable |

Kafka, Hadoop/HDFS, and Kubernetes are explicitly **not** included in the MUST-HAVE tier — they are named only where a genuine function (real-time streaming, true multi-node storage, multi-service scaling) would require them, consistent with the instruction not to include them for decoration.

---

## 7. Technology Stack Selection

| Component | Recommended | Alternative | Reason for Selection |
|---|---|---|---|
| Programming language | Python 3.10+ | Scala | Python has the strongest ML/XAI ecosystem (XGBoost, SHAP, scikit-learn) and is standard for student teams |
| Big Data processing | Apache Spark / PySpark | Dask | Spark is the industry-standard distributed engine; PySpark documentation and community support are extensive |
| Storage format | Parquet | CSV | Columnar, compressed, schema-aware, natively supported by Spark |
| Database | PostgreSQL | SQLite / MongoDB | Relational structure fits prediction logs + metadata; MongoDB adds unneeded complexity for structured, well-typed records |
| Data ingestion | Spark batch read | Kafka (advanced only) | Batch is sufficient for MVP; Kafka reserved for the real-time extension |
| ML framework | XGBoost | LightGBM / Spark MLlib GBT | Best-documented, most widely validated on tabular credit-risk benchmarks (Section 7 of survey below) |
| Explainability | SHAP (TreeExplainer) | LIME | Exact (not approximate) for tree ensembles, faster, and the de facto standard in recent credit-risk XAI literature (Section 18) |
| Backend/API | FastAPI | Flask | Async support, automatic OpenAPI docs, built-in request/response validation via Pydantic |
| Frontend/dashboard | Streamlit | Dash | Fastest path from a Python model to an interactive dashboard for a solo/small-team student project |
| Containerization | Docker | None (bare venv) | Ensures the grader/evaluator can reproduce the exact environment |
| Deployment | Docker Compose (local) → optional single cloud VM | Kubernetes | Compose is sufficient at this scale; Kubernetes is over-engineering for a 2–3 service app |
| Version control | Git + GitHub | GitLab | Standard, free, integrates with most CI options |
| Monitoring (advanced) | Evidently AI / custom drift script | Prometheus+Grafana | Evidently is Python-native and purpose-built for ML data/prediction drift, simpler to add than a full observability stack |

**Official documentation used to verify these capabilities:**
Apache Spark — https://spark.apache.org/docs/latest/ · PySpark — https://spark.apache.org/docs/latest/api/python/ · XGBoost — https://xgboost.readthedocs.io/en/stable/ · LightGBM — https://lightgbm.readthedocs.io/ · SHAP — https://shap.readthedocs.io/en/latest/ · FastAPI — https://fastapi.tiangolo.com/ · Streamlit — https://docs.streamlit.io/ · Docker — https://docs.docker.com/ · PostgreSQL — https://www.postgresql.org/docs/ · Apache Kafka (advanced only) — https://kafka.apache.org/documentation/

---

## 8. Machine Learning Algorithm Survey

| Algorithm | How it works | Strengths | Weaknesses | Scalability | Interpretability | Imbalance handling | Explainability fit |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear decision boundary on (transformed) features | Fast, fully transparent coefficients, regulator-friendly | Misses non-linear interactions common in credit data | Excellent | Very high (native) | Needs class weights | Coefficients are directly interpretable; used as baseline |
| Decision Tree | Recursive feature-threshold splits | Simple to explain visually | High variance, overfits easily | High | High (single tree) | Needs weights/sampling | Easy but unstable explanations |
| Random Forest | Bagged ensemble of decision trees | Robust, less overfitting than a single tree, handles non-linearity | Larger model size, coarser probability calibration | Medium-High | Medium (feature importance only, not per-instance by default) | Class-weight parameter available | SHAP TreeExplainer supported |
| XGBoost | Sequential gradient-boosted trees with regularization | State-of-the-art tabular accuracy, built-in regularization, handles missing values natively, fast (parallelized tree construction) | More hyperparameters to tune; less transparent than logistic regression without SHAP | High (efficient parallel/distributed training) | Medium (needs SHAP for full interpretability) | `scale_pos_weight` parameter, works well with SMOTE | Exact, fast SHAP support (TreeExplainer) |
| LightGBM | Histogram-based, leaf-wise gradient boosting | Faster training and lower memory on large tabular data than XGBoost; strong accuracy | Leaf-wise growth can overfit on smaller data without careful tuning | Very high | Medium (needs SHAP) | `is_unbalance`/`scale_pos_weight` | SHAP-compatible |
| Spark MLlib GBTClassifier | Gradient-boosted trees natively distributed in Spark | Trains directly on Spark DataFrames without exporting data — avoids the driver-memory bottleneck | Fewer tuning knobs and community resources than XGBoost/LightGBM; SHAP support less mature | Highest (true cluster-native) | Medium | Weight column supported | Feature importance native; SHAP integration less mature than XGBoost |

**Research-grounded recommendation:** Multiple recent comparative studies on credit/loan-default data find gradient-boosted tree ensembles (XGBoost, LightGBM, and Gradient Boosting variants) consistently outperforming single-tree and purely linear baselines. <cite index="61-1">One recent Lending Club study reports XGBoost achieving 91.56% accuracy, 94.78% precision, 93.30% recall, and 97.07% AUC, outperforming decision tree, logistic regression, ANN, gradient boosting, and CNN baselines</cite>, while <cite index="60-1">a comparative deep-learning-vs-traditional-ML study on credit risk found resampling-enhanced ensemble models (SMOTE-ENN combined with MLP or random forest) achieving the strongest F1-scores across datasets</cite>. LightGBM is specifically favored when training-time/memory efficiency matters more than marginal accuracy gains, per its design goals in the official LightGBM documentation.

### 8.1 Final Recommendation
1. **Baseline algorithm:** Logistic Regression (with class weighting) — establishes an interpretable, fast lower bound.
2. **Main production/model algorithm:** **XGBoost** — chosen for its strong, widely-replicated accuracy on credit-risk tabular data, native missing-value handling (important given Home Credit's extensive missingness), mature and fast SHAP integration, and the largest body of recent literature validating it specifically for loan-default prediction (Section 18).
3. **Optional comparison algorithm:** LightGBM, to benchmark training-time and accuracy trade-offs, and optionally Spark MLlib GBTClassifier to demonstrate a fully Spark-native training path for the Big Data narrative.

---

## 9. Explainable AI

### 9.1 Method Selection
**SHAP (SHapley Additive exPlanations), using `TreeExplainer`,** is recommended over LIME. TreeExplainer computes exact Shapley values for tree ensembles efficiently (not the local linear approximation LIME uses), and recent credit-risk XAI literature converges on SHAP as the standard: <cite index="31-1">one bank credit-default study combined LightGBM with SHAP specifically because it enables interpretation of the explanatory variables driving predictions, and found this approach reached a level of interpretability comparable to traditional scorecards while preserving the computational advantage of the ML model</cite>. <cite index="34-1">Another recent system combining XGBoost, LightGBM, and Random Forest with both SHAP and LIME for loan-default risk used SMOTE for imbalance and GridSearchCV for tuning</cite>, indicating SHAP is typically paired with, not replacing, sound imbalance-handling and tuning practice.

### 9.2 Global vs. Local Explanation
- **Global**: SHAP summary/beeswarm plot across the validation set shows which features drive risk *overall* (e.g., DTI, `EXT_SOURCE` scores, delinquency count).
- **Local (per-applicant)**: SHAP force/waterfall plot for one prediction, translated into a plain-language card for a loan officer, e.g.:

```
Default probability: 78%

Main contributing (risk-increasing) factors:
+ High debt-to-income ratio
+ Previous late payments
+ High loan-to-income ratio

Risk-reducing factors:
- Stable employment history
- Strong repayment history
```

### 9.3 Explanation ≠ Causation
SHAP values quantify each feature's contribution to *this specific model's output*, not a causal claim about real-world default causation. The dashboard and any officer-facing text must state this explicitly, and the system is positioned throughout this report as **decision-support**, never an autonomous approver/rejecter (Section 13).

---

## 10. Data Preprocessing Pipeline

```
Raw dataset (8 CSVs)
   → Data ingestion (PySpark read, schema inference/enforcement)
   → Schema validation (dtype checks, expected column presence)
   → Missing value treatment (median/mode imputation; missingness itself kept as a feature where informative)
   → Duplicate detection (dedupe on primary keys)
   → Outlier handling (cap/floor extreme AMT_INCOME_TOTAL, DAYS_EMPLOYED anomaly code 365243)
   → Categorical encoding (one-hot for low-cardinality; target/frequency encoding for high-cardinality)
   → Numerical scaling (only for logistic-regression baseline; tree models are scale-invariant)
   → Feature engineering (Section 5.2 derived features; multi-table aggregation)
   → Class imbalance handling (class weights / SMOTE — Section 11)
   → Train/validation/test split (stratified, time-aware if possible)
   → Model training (XGBoost/LightGBM)
   → Evaluation (AUC, PR-AUC, F1, calibration)
   → Explainability (SHAP)
   → Deployment (FastAPI + Docker)
```

**Spark vs. Python/scikit-learn split:** Ingestion, joins, multi-table aggregation, deduplication, and large-scale null handling run in **PySpark** (this is where the Big Data workload actually lives). Once reduced to a single per-applicant feature table (~307K rows × ~150–300 features), it is exported to Pandas/Parquet and the remaining steps — encoding refinement, SMOTE, model training, SHAP, evaluation — run in **Python/scikit-learn/XGBoost/SHAP**, which is both simpler and appropriate once the data is no longer distributed-scale.

---

## 11. Class Imbalance

Loan-default data is inherently imbalanced (~8% positive class in Home Credit, similar orders of magnitude across all three surveyed datasets). Accuracy alone is misleading here: a model predicting "no default" for every applicant would score ~92% accuracy while being useless.

**Recommended methods**, based on recent comparative studies:
- **Class weights** (`scale_pos_weight` in XGBoost) as a first, low-risk baseline.
- **SMOTE / SMOTE-Tomek / SMOTE-ENN** as a should-have enhancement — recent work shows meaningful gains from hybrid resampling: <cite index="58-1">a Boruta+DBSCAN+SMOTE-Tomek+GBM pipeline achieved an F1-score of 82.56%, G-mean of 82.98%, ROC-AUC of 90.90%, and PR-AUC of 91.85% in a credit-default setting</cite>, and <cite index="62-1">an ensemble study across three credit datasets found SMOTE combined with ENN delivered the best overall performance, reaching 90.49% accuracy and 94.61% precision</cite>.
- **Threshold tuning**: rather than the default 0.5 cutoff, choose an operating threshold from the precision-recall curve based on the bank's tolerance for false negatives (missed defaulters) vs. false positives (rejected good applicants).

**Recommended evaluation metrics, prioritized:**
1. **PR-AUC** (most informative under heavy imbalance — focuses on the minority/default class).
2. **ROC-AUC** (standard, comparable across papers/benchmarks).
3. **Recall (of the default class)** — missing an actual defaulter is typically costlier than over-flagging a safe applicant.
4. **Precision** and **F1-score** — balance against excessive false positives (unnecessarily rejecting good applicants).
5. **Calibration/Brier score** — since the output is a *probability* handed to a human decision-maker, well-calibrated probabilities matter more than raw discrimination alone.
6. **Confusion matrix** at the chosen operating threshold, for stakeholder communication.

Accuracy is reported for context only, never as the headline metric.

---

## 12. Add-On Features

| # | Add-on | Category | Technology | Difficulty | In MVP? |
|---|---|---|---|---|---|
| 1 | Real-time loan risk prediction | B | Kafka + Spark Structured Streaming | High | No |
| 2 | Interactive risk dashboard | A | Streamlit | Low | **Yes** |
| 3 | SHAP-based explanation | A | SHAP | Low-Med | **Yes** |
| 4 | Customer risk scoring | A | Model output + DB | Low | **Yes** |
| 5 | Loan approval recommendation | A | Threshold rule on risk score | Low | **Yes** |
| 6 | What-if analysis (slider-based re-scoring) | B | Streamlit widgets + live API call | Medium | Optional (nice stretch) |
| 7 | Customer segmentation | B | K-Means / clustering on feature table | Medium | Optional |
| 8 | Fraud/anomaly detection | C | Isolation Forest / autoencoder | High | No |
| 9 | Fairness/bias analysis | B | Fairlearn / custom group-metric report | Medium | **Should-have** |
| 10 | Model drift monitoring | C | Evidently AI / PSI statistic | Medium-High | No (advanced) |
| 11 | Data drift detection | C | Evidently AI / KS-test on feature distributions | Medium-High | No (advanced) |
| 12 | Automated model retraining | C | Scheduled pipeline (Airflow/cron) | High | No |
| 13 | Risk threshold optimization | B | PR-curve based threshold search | Low-Med | Optional |
| 14 | Loan amount recommendation | C | Constrained optimization on approved risk band | High | No |
| 15 | Early-warning system for existing borrowers | C | Time-series re-scoring on repayment history | High | No |

---

## 13. Real-Time Extension

Kafka is evaluated and **not justified for the MVP**: the project's core value (explainable batch risk scoring for loan applications) does not require sub-second streaming, and applications are naturally submitted one at a time via a form/API call, which a synchronous REST endpoint already serves well.

**If pursued as an advanced extension**, the design is:

```
Loan Application
   → Kafka (application-events topic)
   → Spark Structured Streaming (micro-batch feature computation)
   → Feature Processing (join against precomputed bureau/history aggregates)
   → ML Model (XGBoost, loaded once, applied per micro-batch)
   → Risk Prediction
   → SHAP Explanation (batched)
   → API/Dashboard (push update via websocket or polling)
```

This is explicitly scoped as an **Advanced/stretch** deliverable (Section 21), not a core requirement.

---

## 14. Fairness, Privacy and Security

- **PII**: applicant IDs, addresses, and any name/contact fields (not present in the anonymized Home Credit data, but relevant if a real bank dataset were substituted) must be hashed or dropped before feature engineering.
- **Sensitive attributes**: gender, age, and marital status are present in the raw data; per Section 5.3 these are excluded from model training and evaluated only in a post-hoc fairness audit.
- **Data anonymization**: Home Credit's IDs are already anonymized/synthetic; no re-identification risk in the academic dataset.
- **Encryption**: predictions stored in PostgreSQL should use TLS in transit if deployed beyond localhost; at-rest encryption is a "should-have" for a cloud deployment.
- **Access control**: the API should require a basic API key/auth token even in a demo deployment.
- **Bias & fairness metrics**: recommended checks include **demographic parity difference** and **equalized odds** across the excluded-but-monitored sensitive groups. Recent literature frames this using three canonical criteria: <cite index="57-1">independence (the protected attribute and the predicted outcome should be statistically independent), separation (correlation with the protected attribute is acceptable only to the extent justified by the true outcome, requiring independence within each outcome stratum), and sufficiency (the protected attribute and true outcome should be conditionally independent given the model's score)</cite>. A recent systematic review of the field also cautions that <cite index="56-1">there is no single correct fairness definition — solutions must be tailored to the specific domain, since one-size-fits-all fairness approaches are generally inadequate</cite>.
- **Explainability & transparency**: every prediction shown to a human must be accompanied by its SHAP explanation (Section 9).
- **Human-in-the-loop**: the system is explicitly a **decision-support tool**. The dashboard and API documentation must state that the final approve/reject decision remains with a human loan officer; the model output is one input among several.
- **Data leakage**: care must be taken that no post-outcome fields (e.g., final loan status fields that are only known after default has occurred) leak into training features — this is audited during feature selection.

---

## 15. Deployment Architecture

### 15.1 Component Summary
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Model:** Trained XGBoost model serialized (`.json`/`.ubj` via XGBoost's native format, or `joblib`) and loaded once at API startup
- **Database:** PostgreSQL (prediction logs, model version metadata)
- **Container:** Docker, orchestrated via Docker Compose
- **Deployment:** Local machine / Docker Compose for the MVP; optional single cloud VM (AWS EC2 / GCP Compute Engine / Azure VM free-tier) for public demo access

### 15.2 Data & Prediction Flow
1. A new loan application is submitted (via the Streamlit form or a POST request).
2. FastAPI validates the request payload (Pydantic schema) and computes/derives the same engineered features used in training.
3. The saved XGBoost model produces a default probability.
4. SHAP `TreeExplainer` computes the per-instance explanation for that prediction.
5. The prediction, explanation, and input snapshot are written to PostgreSQL.
6. The API returns `{probability, risk_band, top_factors}` to the caller.
7. The Streamlit dashboard queries the API (or reads directly from PostgreSQL) to render the result and SHAP chart.

### 15.3 Basic Security
API-key header on all POST endpoints; input validation via Pydantic; PostgreSQL credentials via environment variables/Docker secrets, never hard-coded; HTTPS termination if exposed beyond localhost (e.g., via a reverse proxy such as Nginx or a managed platform's built-in TLS).

### 15.4 Local vs. Cloud

| | Local (MVP, required) | Cloud (optional) |
|---|---|---|
| Setup | `docker compose up` on the student's laptop | Same containers deployed to a single small VM |
| Cost | Free | Free-tier eligible on most providers |
| Purpose | Development + grading demo | Public-URL demo for the project defense |

Kubernetes is deliberately **not** recommended — a 3-container Compose stack (API, dashboard, database) does not need cluster orchestration.

---

## 16. System Modules

| Module | Input | Processing | Output | Technology |
|---|---|---|---|---|
| M1 – Data Ingestion | Raw Home Credit CSVs | Schema-enforced Spark read | Raw Spark DataFrames | PySpark |
| M2 – Data Storage | Raw DataFrames | Write partitioned Parquet | Cleaned Parquet lake | Parquet, local disk |
| M3 – Data Preprocessing | Parquet tables | Null/duplicate/outlier handling | Cleaned tables | PySpark |
| M4 – Feature Engineering | Cleaned multi-table data | Joins, aggregations, derived features | Single feature table per applicant | PySpark + Pandas |
| M5 – Model Training | Feature table | Train/val/test split, XGBoost fit, imbalance handling | Trained model artifact | XGBoost, scikit-learn |
| M6 – Model Evaluation | Trained model + test set | Compute AUC/PR-AUC/F1/calibration | Evaluation report | scikit-learn |
| M7 – Explainable AI | Trained model + instance | SHAP value computation | Global/local explanations | SHAP |
| M8 – Risk Prediction API | New application JSON | Feature derivation, model inference | Probability + explanation | FastAPI |
| M9 – Dashboard | API responses / DB | Visualization | Interactive UI | Streamlit |
| M10 – Monitoring (advanced) | Prediction logs | Drift/performance checks | Alerts/reports | Evidently AI (optional) |

---

## 17. API Design

**POST /predict**
Request:
```json
{
  "AMT_INCOME_TOTAL": 202500,
  "AMT_CREDIT": 406597.5,
  "AMT_ANNUITY": 24700.5,
  "DAYS_BIRTH": -9461,
  "DAYS_EMPLOYED": -637,
  "EXT_SOURCE_2": 0.262,
  "...": "remaining engineered features"
}
```
Response:
```json
{
  "default_probability": 0.78,
  "risk_band": "High",
  "recommendation": "Refer for manual review"
}
```

**POST /explain**
Request: same payload as `/predict`.
Response:
```json
{
  "default_probability": 0.78,
  "top_risk_increasing_factors": ["High DTI", "Previous late payments"],
  "top_risk_reducing_factors": ["Stable employment history"],
  "shap_values": { "DTI": 0.14, "EXT_SOURCE_2": -0.09, "...": "..." }
}
```

**GET /health**
Response: `{"status": "ok", "model_version": "xgb_v1.2"}`

**GET /model-info**
Response: `{"model": "XGBoost", "trained_on": "Home Credit Default Risk", "auc": 0.77, "pr_auc": 0.24, "last_trained": "2026-08-01"}`

---

## 18. Dashboard / UI Design

1. **Overview** — portfolio-level default rate, applications processed, model version banner.
2. **Loan Risk Prediction** — form to submit/select an applicant → risk score gauge + risk band.
3. **Individual Customer Explanation** — SHAP waterfall chart, plain-language factor cards (as in Section 9.2).
4. **Model Performance** — ROC curve, PR curve, confusion matrix, calibration plot.
5. **Feature Importance** — global SHAP summary/beeswarm plot.
6. **Data/Default Trends** — default rate by income band, employment type, loan purpose (aggregate charts).
7. **Fairness/Model Monitoring (optional)** — default-rate parity across demographic groups; drift indicator if the advanced monitoring module is implemented.

---

## 19. Technical / Literature Survey

| # | Paper | Authors / Year | Venue | Dataset | Method | Key finding | How it supports this project |
|---|---|---|---|---|---|---|---|
| 1 | Explainable AI for Credit Assessment in Banks | Misheva et al. (as cited in the MDPI JRFM article), 2022 | Journal of Risk and Financial Management (MDPI) | Norwegian bank unsecured consumer loans | LightGBM + SHAP | <cite index="31-1">LightGBM combined with SHAP outperformed the bank's existing logistic-regression scorecard, with credit-balance volatility, remaining-credit percentage, and relationship duration as top predictors</cite> | Validates gradient-boosting + SHAP as a real-bank-deployable combination, directly supporting Sections 8–9 |
| 2 | Credit Risk Prediction Using Explainable AI | 2024 | ResearchGate preprint | Lending Club (P2P) | Decision Tree, LightGBM, Random Forest, XGBoost + SHAP | <cite index="32-1">XGBoost was the most effective model and was further paired with SHAP to explain predictions on P2P lending data</cite> | Directly supports choosing XGBoost as the main model and SHAP as the explainability layer |
| 3 | Explainable AI For Credit Risk Assessment: Integrating Machine Learning With Business Analytics | 2024/2025 | IOSR Journal of Economics and Finance | Multiple / synthesized review | XGBoost, LightGBM, Random Forest + SHAP | <cite index="33-1">Gradient boosting with SHAP produced a comprehensible framework with improved transparency and trustworthiness for credit approval decisions</cite> | Supports the explainability-first framing of Section 9 |
| 4 | (Untitled — arXiv preprint) Explainable AI credit-default system with SMOTE + GridSearchCV | 2025 | arXiv | Not stated in excerpt | XGBoost, LightGBM, Random Forest + SHAP/LIME | <cite index="34-1">Class imbalance handled via SMOTE and hyperparameters tuned with GridSearchCV, evaluated with ROC-AUC, precision, recall, and F1</cite> | Confirms the recommended preprocessing/imbalance/tuning stack in Sections 10–11 |
| 5 | Machine Learning XAI for Early Loan Default Prediction | 2025 | Computational Economics (Springer) | Reviews multiple prior studies (Stevens et al. 2020; Moscato et al. 2021) | XGBoost + SHAP; comparative XAI study | <cite index="36-1">Prior work applying XGBoost with post-hoc SHAP explanations, and a comparative study evaluating several scoring models' explainability using different XAI tools after addressing class imbalance</cite> | Reinforces both algorithm choice and the imbalance-then-explain pipeline order |
| 6 | Big Data Processing for Credit Risk Prediction: An Experimental Study with Apache Spark | 2020 (indexed/cited through 2022+) | Conference paper (ResearchGate) | Three distinct credit datasets | ML on distributed (Spark) vs. non-distributed architectures | <cite index="40-1">Distributed Spark-based architectures became more efficient than non-distributed pipelines as data size increased</cite> | Directly supports the Spark-for-ETL justification in Sections 3 and 6 |
| 7 | Loan Default Prediction Using Spark Machine Learning Algorithms | ~2022 | CEUR Workshop Proceedings | Bank loan dataset | Six Spark ML classifiers (incl. Decision Tree, Random Forest) | <cite index="38-1">Decision Tree and Random Forest models trained via Spark ML achieved the highest accuracy (99.62%) among six classifiers evaluated</cite> | Supports Spark MLlib as a viable native-Spark alternative model path (Section 8) |
| 8 | Performance, Fairness, and Explainability in AI-Based Credit Scoring: A Systematic Literature Review | 2026 | Journal of Risk and Financial Management (MDPI) | Review of 43 papers | Systematic review | <cite index="48-1">Only about 23% of the 43 reviewed papers explicitly measured or proposed bias-mitigation strategies for credit scoring, indicating fairness remains comparatively underexplored relative to explainability and imbalance handling</cite> | Directly motivates the research gap identified in Section 20 |
| 9 | The Fairness of Credit Scoring Models | 2022 (arXiv) | arXiv preprint | Conceptual/methodological | Formal fairness-criteria framework | <cite index="57-1">Defines the independence, separation, and sufficiency fairness criteria used to formally evaluate protected-attribute treatment in credit scores</cite> | Provides the formal fairness vocabulary used in Section 14 |
| 10 | Towards Fair AI: Mitigating Bias in Credit Decisions — A Systematic Literature Review | 2025 | Journal of Risk and Financial Management (MDPI) | Review | Systematic review | <cite index="56-1">Concludes there is no universal fairness definition and that mitigation approaches must be tailored to the specific lending context rather than applied generically</cite> | Supports the context-specific, audited (not one-size-fits-all) fairness approach recommended in Section 14 |
| 11 | Enhancing Credit Default Prediction Using Boruta Feature Selection and DBSCAN with Different Resampling Techniques | 2025 | arXiv | Credit default dataset | Boruta + DBSCAN + SMOTE-Tomek/ADASYN + GBM | <cite index="58-1">The Boruta+DBSCAN+SMOTE-Tomek+GBM pipeline achieved F1 82.56%, G-mean 82.98%, ROC-AUC 90.90%, PR-AUC 91.85%</cite> | Grounds the specific SMOTE-variant recommendations and evaluation metrics in Section 11 |
| 12 | Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction | 2024 | Mathematics (MDPI) | Taiwan, South-German, Belgian credit datasets | Multiple resampling methods + ensemble models | <cite index="62-1">SMOTE combined with Edited Nearest Neighbours (SMOTE+ENN) delivered the best overall performance at 90.49% accuracy, 94.61% precision</cite> | Further validates hybrid resampling recommendations in Section 11 |

*Note on source rigor:* several of the above are recent arXiv preprints or MDPI journal articles rather than IEEE/ACM/Elsevier venues specifically; this reflects the actual current state of publicly retrievable literature on this exact topic and is disclosed transparently rather than misrepresented as top-tier-venue-only. All titles, authors (where stated), years, and findings above are taken directly from the search results retrieved for this report and were not fabricated; a student should re-verify DOIs/full citations directly on the publisher pages before final submission, since some excerpts (e.g., #1, #5, #6, #7) did not expose a full author list or DOI in the retrieved snippet.

---

## 20. Research Gap

| Existing research limitation | Our project's proposed improvement |
|---|---|
| Most credit-XAI papers evaluate SHAP/LightGBM/XGBoost on a **single flat table**, without addressing how explainability pipelines behave on multi-table, Big-Data-scale relational sources like Home Credit's 8-table schema | Build the SHAP explanation layer directly on top of a Spark-engineered multi-table feature set, demonstrating explainability at Big Data scale, not just on a pre-flattened CSV |
| <cite index="48-1">Fairness mitigation is explicitly measured in only ~23% of recent credit-scoring papers</cite>, per a 2026 systematic review | Include a dedicated fairness-audit module (Section 14/12) even though it is optional, rather than omitting it entirely as most surveyed work does |
| Papers on Spark-based credit risk processing (e.g., the Apache Spark experimental study) focus on **training-time scalability**, not on pairing that scalability with **explainability** | Integrate Spark-based ETL/training scalability *and* SHAP-based explainability into one coherent pipeline, rather than treating them as separate research threads |
| Real-time/streaming credit-risk pipelines are rarely evaluated together with explainability in the reviewed literature | Design (even if only as an optional extension) a Kafka+Streaming path that still produces a SHAP explanation per streamed prediction, rather than treating streaming and explainability as mutually exclusive |
| Class-imbalance literature (SMOTE/SMOTE-Tomek/ENN) is well-developed, but rarely combined explicitly with a **decision-support, human-in-the-loop framing** in the same system | Explicitly frame the deployed system as decision-support (not autonomous), pairing sound imbalance handling with a UI that keeps a human loan officer as the final decision-maker |

---

## 21. Project Novelty & Contribution

**Proposed contribution statement:** *An end-to-end, scalable, and explainable loan-risk analytics pipeline that combines distributed multi-table data processing, gradient-boosted machine learning, SHAP-based explanations, and an interactive decision-support dashboard.*

- **Core contribution:** A working, reproducible pipeline from raw multi-table data to an explained, human-reviewable risk decision.
- **Technical contribution:** Spark-based ETL/feature engineering across 8 relational tables, containerized and reproducible via Docker.
- **AI contribution:** Comparative evaluation of Logistic Regression / Random Forest / XGBoost / LightGBM under class-imbalance-aware training.
- **Big Data contribution:** Demonstrated horizontal-scalability architecture (partitioned Parquet, Spark local-mode ETL) that would extend to true cluster deployment without redesign.
- **Explainability contribution:** SHAP-based global and per-applicant local explanations surfaced directly in a loan-officer-facing dashboard.
- **Optional research contribution:** A lightweight fairness audit across excluded sensitive attributes, addressing a gap this survey found in ~77% of recent related papers.

This is presented as an **integration and engineering contribution** (assembling scalability + accuracy + explainability + fairness into one coherent, deployable system), not a claim of a novel algorithm — a realistic and appropriately scoped claim for a final-year project.

---

## 22. MVP vs. Advanced Scope

**MVP**
- Home Credit dataset, Spark-based ETL and feature engineering
- Derived features (DTI, LTI, utilization, delinquency count, etc.)
- XGBoost model with class-weight imbalance handling
- Evaluation via AUC/PR-AUC/F1/confusion matrix
- SHAP global + local explanations
- FastAPI `/predict`, `/explain`, `/health`, `/model-info`
- Streamlit dashboard (Overview, Prediction, Explanation, Model Performance, Feature Importance)
- PostgreSQL prediction logging
- Docker Compose deployment
- Git/GitHub with documented README

**Advanced (stretch goals, time permitting)**
- SMOTE/SMOTE-Tomek imbalance handling and LightGBM comparison model
- Kafka + Spark Structured Streaming real-time scoring path
- Fairness/bias audit dashboard page
- Basic drift-detection script (Evidently AI, run manually, not fully automated)
- Automated retraining trigger
- Cloud VM deployment for public demo access

---

## 23. Development Roadmap

| Phase | Focus | Deliverables |
|---|---|---|
| 1 | Dataset & literature | Dataset downloaded/verified, literature review draft, this feasibility report finalized |
| 2 | Data engineering | PySpark ingestion, schema validation, Parquet lake for all 8 tables |
| 3 | Feature engineering | Joined/aggregated single feature table, derived features implemented and unit-tested |
| 4 | Baseline ML | Logistic Regression baseline, initial XGBoost run, baseline metrics report |
| 5 | Model optimization | Hyperparameter tuning, imbalance handling (class weights, then SMOTE), LightGBM comparison |
| 6 | Explainable AI | SHAP integration, global/local explanation functions, plain-language factor generator |
| 7 | Backend/API | FastAPI endpoints, request validation, model-loading service |
| 8 | Dashboard | Streamlit pages per Section 18, wired to the API |
| 9 | Deployment | Dockerfiles, Docker Compose, PostgreSQL integration, (optional) cloud VM |
| 10 | Testing/documentation | End-to-end test run, README, architecture diagram, final report/slides, project defense prep |

---

## 24. Risks & Mitigation
See Section 3.7 for the consolidated risk register (environment setup, memory pressure on large tables, class imbalance, SHAP performance, scope creep).

---

## 25. Final Feasibility Verdict

| Dimension | Rating |
|---|---|
| Technical feasibility | **High** |
| Data feasibility | **High** |
| Computational feasibility | **High** (local machine sufficient) |
| Implementation complexity | **Medium** |
| Research potential | **Medium-High** (fairness-gap angle gives genuine novelty room) |
| Final-year suitability | **High** |

**Final recommended architecture/stack:** Home Credit Default Risk dataset → PySpark ETL/feature engineering → XGBoost (class-weighted, LightGBM as comparison) → SHAP explanations → FastAPI backend → Streamlit dashboard → PostgreSQL logging → Docker Compose deployment, with Kafka/streaming, fairness auditing, and cloud deployment retained as clearly-labeled optional extensions.

---

## 26. Official Technical Documentation

- Apache Spark — https://spark.apache.org/docs/latest/
- PySpark API — https://spark.apache.org/docs/latest/api/python/
- XGBoost — https://xgboost.readthedocs.io/en/stable/
- LightGBM — https://lightgbm.readthedocs.io/
- SHAP — https://shap.readthedocs.io/en/latest/
- scikit-learn — https://scikit-learn.org/stable/
- FastAPI — https://fastapi.tiangolo.com/
- Streamlit — https://docs.streamlit.io/
- Docker — https://docs.docker.com/
- PostgreSQL — https://www.postgresql.org/docs/
- Apache Kafka (advanced extension only) — https://kafka.apache.org/documentation/
- Home Credit Default Risk dataset (Kaggle) — https://www.kaggle.com/c/home-credit-default-risk/data
- Lending Club dataset (Kaggle) — https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Give Me Some Credit dataset (Kaggle) — https://www.kaggle.com/c/give-me-some-credit

## 27. Research References

1. Explainable AI for Credit Assessment in Banks — Journal of Risk and Financial Management (MDPI), 2022. https://www.mdpi.com/1911-8074/15/12/556
2. Credit Risk Prediction Using Explainable AI — 2024. https://www.researchgate.net/publication/379068563_Credit_Risk_Prediction_Using_Explainable_AI
3. Explainable AI For Credit Risk Assessment: Integrating Machine Learning With Business Analytics — IOSR Journal of Economics and Finance. https://www.iosrjournals.org/iosr-jef/papers/Vol16-Issue5/Ser-1/H1605016372.pdf
4. Explainable Artificial Intelligence Credit Risk Assessment (XGBoost/LightGBM/RF + SHAP/LIME, SMOTE, GridSearchCV) — arXiv, 2025. https://arxiv.org/pdf/2506.19383
5. Machine Learning XAI for Early Loan Default Prediction — Computational Economics (Springer), 2025. https://link.springer.com/article/10.1007/s10614-025-10962-9
6. Big Data Processing for Credit Risk Prediction: An Experimental Study with Apache Spark. https://www.researchgate.net/publication/345976972_Big_Data_Processing_for_Credit_Risk_Prediction_An_Experimental_Study_with_Apache_Spark_K
7. Loan Default Prediction Using Spark Machine Learning Algorithms — CEUR Workshop Proceedings. https://ceur-ws.org/Vol-3105/paper30.pdf
8. Performance, Fairness, and Explainability in AI-Based Credit Scoring: A Systematic Literature Review — Journal of Risk and Financial Management (MDPI), 2026. https://www.mdpi.com/1911-8074/19/2/104
9. The Fairness of Credit Scoring Models — arXiv, 2022. https://arxiv.org/pdf/2205.10200
10. Towards Fair AI: Mitigating Bias in Credit Decisions — A Systematic Literature Review — Journal of Risk and Financial Management (MDPI), 2025. https://www.mdpi.com/1911-8074/18/5/228
11. Enhancing Credit Default Prediction Using Boruta Feature Selection and DBSCAN Algorithm with Different Resampling Techniques — arXiv, 2025. https://arxiv.org/html/2509.19408
12. Ensemble-Based Machine Learning Algorithm for Loan Default Risk Prediction — Mathematics (MDPI), 2024. https://www.mdpi.com/2227-7390/12/21/3423

**Important caveat for the student:** items 1, 5, 6, and 7 above were retrieved via secondary excerpts that did not expose complete author lists or DOIs. Before final submission, open each URL directly and copy the full citation (authors, exact title, DOI) from the publisher's page itself, since this report intentionally does not fabricate any author name or DOI that was not directly visible in the retrieved source.

---

*End of report.*
