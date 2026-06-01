# Reproducibility Steps & Theoretical Mapping

This document provides a technical walkthrough connecting the Python pipeline implemented in this repository directly to the theoretical framework and hypotheses presented in the paper.

## Phase 1: Data Integration & Engineering (H1 Validation)

**Script:** `step1_2_data_processing_v2.py` & Initial steps of `step3_4_5_pipeline_v2.py`

1.  **Addressing Information Asymmetry**: The pipeline automatically merges traditional application data with Alternative Behavioral Data.
2.  **Key Engineered Features**:
    - `INST_UNDERPAY_MEAN`: Mean installment underpayment in VND. (Identified by SHAP as the top tail-risk indicator).
    - `INST_PAY_DELAY_MEAN`: Mean payment delay in days.
    - `AIR` & `DIR`: Annuity-to-Income and Debt-to-Income ratios, crucial for assessing repayment capacity in markets with high informal income.
    - `EXT_SOURCES_PROD`: Multiplicative interactions of standardized (300-850) external bureau scores.

## Phase 2: Algorithm Benchmarking

**Script:** `step3_4_5_pipeline_v2.py` (Cross-Validation Section)

1.  **Execution**: The script runs a 3-Fold Stratified Cross-Validation to maintain the operational ~92%/8% class imbalance.
2.  **Algorithms Evaluated**: Random Forest, LightGBM, XGBoost, and CatBoost.
3.  **Metrics Captured**: AUC-ROC (Discriminative power), Gini, KS Statistic, and Brier Score (Calibration).
4.  **Paper Alignment**: As detailed in Section IV.A, CatBoost emerges as the optimal architecture for this specific dataset scale (15,000 training records) and feature heterogeneity, outperforming LightGBM which typically excels only on massive-scale datasets.

## Phase 3: Cost-Sensitive Optimization & EFL (H2 Validation)

**Script:** `step3_4_5_pipeline_v2.py` (Cost-Sensitive Section)

1.  **The Optimization**: The script isolates the best-performing model (CatBoost) and retrains it using a modified loss function: `scale_pos_weight = 11.5`.
2.  **Economic Evaluation**: The script dynamically calculates the Expected Financial Loss (EFL).
    - It captures the shift in the confusion matrix (drastically increasing True Positives / caught defaults, while accepting higher False Positives).
    - It calculates the theoretical financial savings (e.g., 3,520 million VND) by preventing expensive False Negatives.
3.  **Calibration Trade-off**: The script logs the Brier Score deterioration, demonstrating the theoretical trade-off between financial optimization and pure statistical probability calibration.

## Phase 4: Model Interpretability (XAI)

**Script:** `step3_4_5_pipeline_v2.py` (SHAP Section)

1.  **Execution**: Uses `shap.TreeExplainer` on a 500-applicant subsample.
2.  **Outputs**: Generates global feature importance plots and SHAP Summary Plots.
3.  **Regulatory Compliance**: These outputs fulfill the interpretability requirements for Vietnamese banking regulations (and Basel III), proving that decisions are based on coherent behavioral signals rather than opaque artifacts.

## Phase 5: Production Inference

**Script:** `predict_new_customers_v2.py`

1.  **Simulated Deployment**: Demonstrates how the cost-sensitive model would be deployed in a real-world Loan Origination System.
2.  **Threshold Adjustment**: Because `scale_pos_weight` pushes predicted probabilities upward to catch borderline defaults, the operational approval threshold must be adjusted accordingly (e.g., shifting the conceptual 15% risk tolerance to a 65% model threshold). The script implements this logic to yield final `Approve/Reject` decisions.
