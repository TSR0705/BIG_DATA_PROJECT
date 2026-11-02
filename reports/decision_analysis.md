# Experiment E4: Decision Analysis & Cost-Sensitive Optimization Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 10 — Research Experiments (E4)  
**Execution Timestamp:** 2026-11-02 17:34:44 UTC  
**Evaluator Model:** Frozen Phase 8 Finalist XGBoost (`max_depth=3`, `learning_rate=0.1`, `n_estimators=500`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=1.0`)  
**Data Partition:** Validation ($N=46,127$, positive defaults = 3,724). **Zero test data accessed.**  

---

## 1. Executive Summary & Problem Formulation

In retail credit underwriting, the cost of a **False Negative** ($C_\text{FN}$: approving an applicant who subsequently defaults) drastically exceeds the cost of a **False Positive** ($C_\text{FP}$: declining a creditworthy applicant, incurring only lost origination fee/interest spread).

Consequently, standard statistical cutoffs (such as $\tau = 0.50$) are economically sub-optimal. Under an asymmetric error structure, total decision cost across $N$ applicants is computed as:

$$\text{Total Cost} = \text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}}$$

$$\text{Cost per Applicant} = \frac{\text{Total Cost}}{N}$$

where $C_{\text{FN}} : C_{\text{FP}}$ represents the relative loss severity ratio between a defaulted loan charge-off and an unnecessary customer decline.

This experiment investigates:
1. **Empirical vs. Bayesian Optimal Thresholds:** Quantifying exact decision boundaries under 5:1, 10:1, and 20:1 cost asymmetries.
2. **Economic Value of Calibration:** Determining the cost reduction achieved by optimizing $\tau$ relative to default cutoffs.
3. **Risk Segmentation Viability:** Validating a practical 3-tier risk band strategy (LOW, MEDIUM, HIGH) for operational credit policy.

---

## 2. Cost-Sensitive Threshold Optimization Results

### Table 2.1: Optimal Operating Points Across Asymmetric Error Cost Ratios

| Error Cost Ratio ($C_\text{FN} : C_\text{FP}$) | Empirical Optimal $\tau^*$ | Bayesian Theoretical $\tau$ | Min Total Cost | Cost per Applicant | Cost at $\tau=0.50$ | Cost Savings vs. 0.50 | Confusion Matrix (TP, FP, TN, FN) | Approval Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5:1** | **0.16** | 0.1667 | 14,364 | 0.3114 | 17,791 | **19.26%** | TP=1,743, FP=4,459, TN=37,944, FN=1,981 | 86.55% |
| **10:1** | **0.09** | 0.0909 | 22,359 | 0.4847 | 35,446 | **36.92%** | TP=2,543, FP=10,549, TN=31,854, FN=1,181 | 71.62% |
| **20:1** | **0.06** | 0.0476 | 30,475 | 0.6607 | 70,756 | **56.93%** | TP=3,024, FP=16,475, TN=25,928, FN=700 | 57.73% |

> [!NOTE]
> All costs are expressed in relative decision-loss units normalized to $C_{\text{FP}} = 1.0$, rather than absolute currency denominations, ensuring methodological invariance across portfolio loan balances.
> The full 99-step threshold sweep is persisted in [reports/decision_threshold_analysis.csv](file:///C:/Users/ACER/Desktop/BIG_DATA_PROJECT/reports/decision_threshold_analysis.csv), keeping the frozen Phase 8 baseline artifact [reports/threshold_analysis.csv](file:///C:/Users/ACER/Desktop/BIG_DATA_PROJECT/reports/threshold_analysis.csv) untouched.

---

## 3. Three-Tier Credit Risk Band Segmentation & Monotonicity Audit

To support operational underwriting workflows without requiring a single hard binary cutoff, applicants are partitioned into three distinct risk tiers:

### Table 3.1: Risk Band Profile and Ordinal Monotonicity Verification

| Risk Band | Probability Range | Applicant Count | Population Share (%) | Defaults | Observed Default Rate (%) | Mean Predicted Risk (%) | Recommended Policy Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** | `[0.00, 0.09)` | 33,035 | 71.62% | 1,181 | **3.57%** | 3.77% | Fast-track Approval / Prime Rate |
| **MEDIUM** | `[0.09, 0.25)` | 10,258 | 22.24% | 1,513 | **14.75%** | 14.53% | Manual Underwriting Review / Standard Rate |
| **HIGH** | `[0.25, 1.00)` | 2,834 | 6.14% | 1,030 | **36.34%** | 36.67% | Decline / Require Collateral / Strict Underwriting |

### Monotonicity Verification:
- **LOW Tier:** 3.57% observed default rate
- **MEDIUM Tier:** 14.75% observed default rate
- **HIGH Tier:** 36.34% observed default rate

$$\text{Default Rate}(\text{LOW}) < \text{Default Rate}(\text{MEDIUM}) < \text{Default Rate}(\text{HIGH}) \implies \mathbf{3.57\% < 14.75\% < 36.34\%}$$

> [!NOTE]
> In credit risk portfolio segmentation, strict monotonicity ensures that higher risk bands reflect higher empirical default rates ($3.57\% < 14.75\% < 36.34\%$). While not a substitute for formal regulatory capital model validation, this confirms proper ordinal calibration across operational policy segments.

---

## 4. Key Takeaways for Credit Policy Implementation

1. **Substantial Cost Reduction:** Moving from the naive $\tau = 0.50$ threshold to cost-optimized thresholds reduces financial misclassification losses by **19.3% to 56.9%** depending on the loss severity ratio.
2. **Alignment with Theoretical Boundaries:** The empirical optimal thresholds closely match Bayesian decision theory predictions ($\tau^* \approx 0.16$ vs theoretical 0.1667 for 5:1; $\tau^* \approx 0.09$ vs theoretical 0.0909 for 10:1; $\tau^* \approx 0.06$ vs theoretical 0.0476 for 20:1).
3. **Operational Policy Integration:**
   - 71.6% of applicants fall into the LOW risk band and can receive automated prime-rate approval.
   - 22.2% of applicants fall into the MEDIUM band and are routed to manual income verification.
   - 6.1% fall into HIGH risk, preventing severe portfolio charge-offs.
