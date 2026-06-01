# Cost-Sensitive Boosting Benchmark for Credit Risk Assessment

This repository contains the official code implementation, data processing pipeline, and reproducible benchmark framework for the paper: **"Benchmarking Cost-Sensitive Boosting Algorithms for Credit Risk Assessment using Alternative Behavioral Data: A Case Study in Vietnamese Consumer Finance"**.

This project provides a complete machine learning pipeline to evaluate default risk (Credit Scoring) using both traditional application data and alternative behavioral data (installment payments, previous applications, and external bureau scores).

---

## 1. Project Structure

To ensure a clean, reproducible, and anonymous double-blind review process, the repository is structured as follows:

*   **`Input2/`**: *(Ignored in git)* Directory for the raw operational data files (e.g., `ho_so_khach_hang_train.csv`, `lich_su_tin_dung_bureau.csv`).
*   **`data_output_v2/`**: *(Ignored in git)* Directory for the fused and cleaned 24-feature dataset (`cleaned_data2.csv`).
*   **`new_customers/`**: Contains templates and small synthetic datasets for inference testing.
    *   `template.csv`: A minimal structural template.
    *   `synthetic_new_customers_50.csv`: A synthetic dataset provided for reviewers to test the prediction pipeline without requiring the proprietary dataset.
*   **`results2/`** & **`plots2/`**: *(Generated during runtime)* Directories where the pipeline automatically saves evaluation metrics, Brier scores, Expected Financial Loss (EFL) logs, and SHAP visualizations.
*   **Core Pipeline Scripts**:
    *   `step1_2_data_processing_v2.py`: Data fusion and feature engineering.
    *   `step3_4_5_pipeline_v2.py`: The main benchmarking, cross-validation, and cost-sensitive training script.
    *   `predict_new_customers_v2.py`: Inference script for evaluating new applicants.
*   **`data_update_guide.md`**: Data schema guidelines for institutional deployment.

---

## 2. Reproducibility Guide: 3-Step Execution Pipeline

Reviewers can reproduce the experimental pipeline by executing the following scripts sequentially. Ensure that all dependencies listed in `requirements.txt` are installed (`pip install -r requirements.txt`).

### STEP 1: Data Fusion & Feature Engineering
*   **Command**:
    ```bash
    python step1_2_data_processing_v2.py
    ```
*   **Process**:
    *   Loads the primary applicant dataset.
    *   Aggregates external bureau records (extracting mean credit durations and counts).
    *   Aggregates previous internal loan applications (calculating rejection rates and mean loan amounts).
    *   Aggregates installment behaviors (extracting `INST_PAY_DELAY_MEAN` and `INST_UNDERPAY_MEAN`).
    *   Fuses these sources into a consolidated dataset (`data_output_v2/cleaned_data2.csv`).

### STEP 2: Algorithm Benchmarking, Cost-Sensitive Optimization, and XAI
*   **Command**:
    ```bash
    python step3_4_5_pipeline_v2.py
    ```
*   **Process**:
    *   **Feature Engineering Phase II**: Derives key financial ratios (`DIR`, `AIR`, `ACR`, `DAR`) and bureau score interactions (`EXT_SOURCES_PROD`).
    *   **Benchmarking**: Evaluates Random Forest, LightGBM, XGBoost, and CatBoost under **3-Fold Stratified Cross-Validation**. Computes AUC-ROC, Gini Coefficient, KS Statistic, and Brier Score.
    *   **Cost-Sensitive Optimization**: Retrains the optimal base learner (CatBoost) applying a minority-class weight scaling (`scale_pos_weight = 11.5`) to address the ~8% default class imbalance.
    *   **Economic Evaluation**: Calculates the Expected Financial Loss (EFL) savings between the baseline and the cost-sensitive model.
    *   **Explainable AI (XAI)**: Generates feature importance rankings and calculates game-theoretic **SHAP values**. Outputs `feature_importance_v2.png` and `shap_summary_v2.png`.
    *   *Note: Logs and plots are saved to the automatically generated `final_result/` (or `MinhChung...`) directories.*

### STEP 3: Inference and Decisioning for New Applicants
*   **Command**:
    ```bash
    python predict_new_customers_v2.py
    ```
*   **Process**:
    *   The system prompts for a path to a new customer CSV file (Reviewers can press Enter to use the default synthetic dataset).
    *   Loads the pre-trained LightGBM/CatBoost model.
    *   Executes the feature engineering pipeline dynamically on the new records.
    *   Outputs the calibrated default probability and makes a final `Approve/Reject` decision based on a defined risk threshold (adjusted for the cost-sensitive shift).
    *   Generates a final evaluation report in the `results2/` directory.

---

## 3. Theoretical Framework & Evaluation Metrics

The experimental framework implemented in this repository is designed to bridge statistical machine learning metrics with practical financial risk operations:

*   **Expected Financial Loss (EFL):**
    $$EFL = (FN \times D) + (FP \times C)$$
    Where $D$ is the principal exposure of a missed default (False Negative) and $C$ is the opportunity cost of a false rejection (False Positive).

*   **Brier Score (Probability Calibration):**
    $$Brier = \frac{1}{N} \sum_{i=1}^{N} (y_i - p_i)^2$$
    Tracks the deliberate probability miscalibration introduced by cost-sensitive learning to minimize EFL.

*   **SHAP (Shapley Additive Explanations):**
    Provides individual-level, regulatory-compliant explanations for adverse action notices, identifying `INST_UNDERPAY_MEAN`, `DIR`, and `AIR` as dominant risk drivers in the Vietnamese market context.
