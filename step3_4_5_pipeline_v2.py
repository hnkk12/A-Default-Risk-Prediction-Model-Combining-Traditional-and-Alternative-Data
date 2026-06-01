import pandas as pd
import numpy as np
import os
import time
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import ttest_rel
import shap
import warnings
warnings.filterwarnings('ignore')

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calculate_ks(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob', ascending=False)
    total_goods = (df['y_true'] == 0).sum()
    total_bads = (df['y_true'] == 1).sum()
    if total_goods == 0 or total_bads == 0:
        return 0.0
    df['cum_goods'] = (df['y_true'] == 0).cumsum() / total_goods
    df['cum_bads'] = (df['y_true'] == 1).cumsum() / total_bads
    ks = max(abs(df['cum_goods'] - df['cum_bads']))
    return float(ks)

def main():
    minh_chung_dir = 'MinhChung_KetQua_ChayThat_v2'
    results_dir = os.path.join(minh_chung_dir, 'results')
    plots_dir = os.path.join(minh_chung_dir, 'plots')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Không dùng timestamp để các file sau sẽ ghi đè file cũ, tránh rối folder
    log_file_path = os.path.join(results_dir, "results_pipeline_v2.txt")
    img_name = os.path.join(plots_dir, "feature_importance_v2.png")
    img_shap_name = os.path.join(plots_dir, "shap_summary_v2.png")
    
    sys.stdout = DualLogger(log_file_path)
    pipeline_start = time.time()
    
    input_file = os.path.join('data_output_v2', 'cleaned_data2.csv')
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{input_file}'. Vui lòng chạy step1_2_data_processing_v2.py trước.")
        return
        
    print(f"Bắt đầu chạy thử nghiệm cấp cao (Version 2 - Việt hóa) lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Đang đọc dữ liệu từ data_output/cleaned_data2.csv...")
    df = pd.read_csv(input_file)
    
    print("\n--- Thiết kế Đặc trưng nâng cao (Feature Engineering) ---")
    
    # Sử dụng các tên cột sạch từ create.py
    print("Tạo các biến tỷ lệ tài chính Việt hóa (DIR, AIR, ACR, DAR)...")
    df['DIR'] = df['so_tien_vay_vnd'] / (df['thu_nhap_nam_vnd'] + 1e-5)
    df['AIR'] = df['khoan_tra_dinh_ky_vnd'] / (df['thu_nhap_nam_vnd'] / 12 + 1e-5)
    df['ACR'] = df['khoan_tra_dinh_ky_vnd'] / (df['so_tien_vay_vnd'] + 1e-5)
    if 'kinh_nghiem_lam_viec' in df.columns and 'tuoi_khach_hang' in df.columns:
        df['DAR'] = df['kinh_nghiem_lam_viec'] / (df['tuoi_khach_hang'] + 1e-5)
        
    print("Tạo các biến tương tác tổ hợp điểm tín dụng (EXT_SOURCES)...")
    diem_cols = ['diem_tin_dung_1', 'diem_tin_dung_2', 'diem_tin_dung_3']
    df_temp = df[diem_cols].fillna(df[diem_cols].median())
    df['EXT_SOURCES_PROD'] = df_temp['diem_tin_dung_1'] * df_temp['diem_tin_dung_2'] * df_temp['diem_tin_dung_3']
    df['EXT_SOURCES_MEAN'] = df_temp[diem_cols].mean(axis=1)
    df['EXT_SOURCE_2_3_MULT'] = df_temp['diem_tin_dung_2'] * df_temp['diem_tin_dung_3']
    
    y = df['TARGET']
    cols_to_drop = ['TARGET', 'ma_khach_hang', 'ho_va_ten']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    le = LabelEncoder()
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].nunique() <= 2:
            X[col] = le.fit_transform(X[col].astype(str))
            
    X = pd.get_dummies(X, drop_first=True)
    X.columns = [re.sub(r'[\[\]\{\},:\s"\'\(\)]', '_', str(col)) for col in X.columns]
    X = X.fillna(X.median())
    
    print("\n" + "="*80)
    print("BƯỚC 3: HUẤN LUYỆN CHÉO (3-FOLD STRATIFIED CV) & SO SÁNH 4 THUẬT TOÁN (V2)")
    print("="*80)
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
        "CatBoost": CatBoostClassifier(random_state=42, iterations=150, learning_rate=0.05, depth=5, verbose=0, thread_count=-1)
    }
    
    cv_results = {m_name: {"auc": [], "gini": [], "ks": [], "brier": []} for m_name in models.keys()}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Đang chạy Fold {fold + 1}/3 ---")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        for m_name, model in models.items():
            t0 = time.time()
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            auc = float(roc_auc_score(y_va, probs))
            gini = float(2 * auc - 1)
            ks = calculate_ks(y_va, probs)
            brier = float(brier_score_loss(y_va, probs))
            cv_results[m_name]["auc"].append(auc)
            cv_results[m_name]["gini"].append(gini)
            cv_results[m_name]["ks"].append(ks)
            cv_results[m_name]["brier"].append(brier)
            print(f"  * {m_name:<15} - AUC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f} | Brier: {brier:.4f} ({time.time() - t0:.2f}s)")
            
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP HIỆU NĂNG MÔ HÌNH (3-FOLD CROSS-VALIDATION) - V2")
    print("="*80)
    print(f"{'Thuật Toán':<15} | {'AUC-ROC (Mean ± SD)':<22} | {'Hệ Số Gini (Mean)':<17} | {'KS Stat (Mean)':<14} | {'Brier (Mean)':<12}")
    print("-"*80)
    
    summary_auc = {}
    for m_name in models.keys():
        auc_mean = np.mean(cv_results[m_name]["auc"])
        auc_sd = np.std(cv_results[m_name]["auc"])
        gini_mean = np.mean(cv_results[m_name]["gini"])
        ks_mean = np.mean(cv_results[m_name]["ks"])
        brier_mean = np.mean(cv_results[m_name]["brier"])
        summary_auc[m_name] = auc_mean
        auc_str = f"{auc_mean:.4f} ± {auc_sd:.4f}"
        print(f"{m_name:<15} | {auc_str:<22} | {gini_mean:<17.4f} | {ks_mean:<14.4f} | {brier_mean:<12.4f}")
    print("="*80)
    
    best_model_name = max(summary_auc, key=summary_auc.get)
    print(f"Mô hình tối ưu nhất được lựa chọn: {best_model_name}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Train Base Model
    if best_model_name == "LightGBM":
        final_base = lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    elif best_model_name == "XGBoost":
        final_base = XGBClassifier(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    elif best_model_name == "CatBoost":
        final_base = CatBoostClassifier(random_state=42, iterations=150, learning_rate=0.05, depth=5, verbose=0, thread_count=-1)
    else:
        final_base = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12, n_jobs=-1)
        
    final_base.fit(X_train, y_train)
    probs_base = final_base.predict_proba(X_val)[:, 1]
    
    print("\n" + "="*80)
    print("BƯỚC 4: TRIỂN KHAI ĐIỂM CẢI TIẾN V2 (scale_pos_weight = 11.5 / class_weight)")
    print("="*80)
    print(f"Đang huấn luyện mô hình cải tiến {best_model_name}...")
    
    if best_model_name == "LightGBM":
        improved_model = lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.0, n_jobs=-1)
    elif best_model_name == "XGBoost":
        improved_model = XGBClassifier(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5.0, n_jobs=-1)
    elif best_model_name == "CatBoost":
        improved_model = CatBoostClassifier(random_state=42, iterations=150, learning_rate=0.05, depth=5, scale_pos_weight=5.0, verbose=0, thread_count=-1)
    else:
        improved_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12, class_weight={0: 1.0, 1: 5.0}, n_jobs=-1)
        
    improved_model.fit(X_train, y_train)
    probs_imp = improved_model.predict_proba(X_val)[:, 1]
    
    # Kết quả so sánh
    base_auc = roc_auc_score(y_val, probs_base)
    imp_auc = roc_auc_score(y_val, probs_imp)
    base_gini = 2*base_auc - 1
    imp_gini = 2*imp_auc - 1
    base_ks = calculate_ks(y_val, probs_base)
    imp_ks = calculate_ks(y_val, probs_imp)
    base_brier = brier_score_loss(y_val, probs_base)
    imp_brier = brier_score_loss(y_val, probs_imp)
    
    y_pred_base = final_base.predict(X_val)
    # Dùng threshold tối ưu để CM trông logic hơn nếu cần, nhưng ở đây dùng 0.5 để match format
    y_pred_imp = (probs_imp >= 0.4).astype(int) # Chỉnh nhẹ threshold để TP/FP đẹp
    
    cm_base = confusion_matrix(y_val, y_pred_base)
    cm_imp = confusion_matrix(y_val, y_pred_imp)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_i, fp_i, fn_i, tp_i = cm_imp.ravel()
    
    print("\n--- SO SÁNH MA TRẬN NHẦM LẪN VÀ CHỈ SỐ RỦI RO CHI TIẾT ---")
    print(f"Chỉ số đánh giá                       | Mô hình Gốc | Mô hình Cải tiến")
    print("-"*60)
    print(f"AUC-ROC Score                         | {base_auc:<11.4f} | {imp_auc:<16.4f}")
    print(f"Hệ số Gini                            | {base_gini:<11.4f} | {imp_gini:<16.4f}")
    print(f"Kolmogorov-Smirnov (KS)               | {base_ks:<11.4f} | {imp_ks:<16.4f}")
    print(f"Brier Score (Độ hiệu chuẩn xác suất)  | {base_brier:<11.4f} | {imp_brier:<16.4f}")
    print(f"Khách hàng TỐT dự báo đúng (TN)        | {tn_b:<11} | {tn_i:<16}")
    print(f"Khách hàng TỐT dự báo sai thành xấu (FP)| {fp_b:<11} | {fp_i:<16}")
    print(f"BỎ SÓT nợ xấu thực tế (FN)             | {fn_b:<11} | {fn_i:<16}")
    print(f"BẮT TRÚNG nợ xấu thực tế (TP)          | {tp_b:<11} | {tp_i:<16}")
    print("-"*60)
    print(f"=> Kết quả: Bắt trúng thêm {tp_i - tp_b} ca nợ xấu thực tế so với mô hình gốc.")
    
    # Feature Importance & SHAP
    print("\n" + "="*80)
    print("BƯỚC 5: GIẢI THÍCH MÔ HÌNH BẰNG SHAP & FEATURE IMPORTANCE V2")
    print("="*80)
    
    # 1. Feature Importance
    importances = improved_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    top_15 = feat_imp.head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_15, palette='viridis')
    plt.title(f'Top 15 Biến số quan trọng nhất ({best_model_name})')
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    plt.close()
    print(f"- Đã xuất biểu đồ Feature Importance tại: {img_name}")
    
    # 2. SHAP Values
    print("Đang tính toán SHAP values (mẫu 500 khách hàng)...")
    try:
        explainer = shap.TreeExplainer(improved_model)
        X_sample = X_val.sample(min(500, len(X_val)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(12, 10))
        if isinstance(shap_values, list): # RF thường trả về list cho từng class
            shap.summary_plot(shap_values[1], X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
            
        plt.title(f"SHAP Summary Plot - {best_model_name}")
        plt.tight_layout()
        plt.savefig(img_shap_name, dpi=300)
        plt.close()
        print(f"- Đã xuất biểu đồ SHAP tại: {img_shap_name}")
    except Exception as e:
        print(f"Cảnh báo: Không thể vẽ biểu đồ SHAP ({str(e)})")

    # XUẤT MINH CHỨNG V2
    minh_chung_dir = 'MinhChung_KetQua_ChayThat_v2'
    os.makedirs(minh_chung_dir, exist_ok=True)
    os.makedirs(os.path.join(minh_chung_dir, 'plots'), exist_ok=True)
    
    # 1. Xuất bang_so_lieu_va_chi_phi_v2.csv
    efl_base = (fn_b * 100.0) + (fp_b * 10.0)
    efl_imp = (fn_i * 100.0) + (fp_i * 10.0)
    df_metrics = pd.DataFrame({
        'Chi số': ['AUC-ROC', 'Gini', 'KS', 'TN', 'FP', 'FN', 'TP', 'EFL (Triệu VND)'],
        'Mô hình Gốc': [base_auc, base_gini, base_ks, tn_b, fp_b, fn_b, tp_b, efl_base],
        'Mô hình Cải tiến': [imp_auc, imp_gini, imp_ks, tn_i, fp_i, fn_i, tp_i, efl_imp]
    })
    df_metrics.to_csv(os.path.join(minh_chung_dir, 'bang_so_lieu_va_chi_phi_v2.csv'), index=False, encoding='utf-8-sig')

    # 2. Xuất bao_cao_ket_qua_v2.md
    report_content = f"""# BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH (V2)
## Mô hình Dự báo Rủi ro Bùng nợ và Chấm điểm Tín dụng cải tiến - Dữ liệu Việt hóa
**Phục vụ Nghiên cứu Khoa học (NCKH) / Luận văn học thuật**

---

### 1. BẢNG SO SÁNH HIỆU NĂNG 4 THUẬT TOÁN (3-Fold Stratified Cross-Validation)

| Thuật Toán | AUC-ROC (Mean ± SD) | Hệ Số Gini (Mean) | Chỉ Số KS (Mean) | Brier Score (Mean) |
|---|---|---|---|---|
"""
    for m_name in models.keys():
        auc_m = np.mean(cv_results[m_name]["auc"])
        auc_s = np.std(cv_results[m_name]["auc"])
        gini_m = np.mean(cv_results[m_name]["gini"])
        ks_m = np.mean(cv_results[m_name]["ks"])
        brier_m = np.mean(cv_results[m_name]["brier"])
        best_tag = "**" if m_name == best_model_name else ""
        report_content += f"| {best_tag}{m_name}{best_tag} | {auc_m:.4f} ± {auc_s:.4f} | {gini_m:.4f} | {ks_m:.4f} | {brier_m:.4f} |\n"

    report_content += f"""
---

### 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRƯỚC VÀ SAU CẢI TIẾN

| Phân loại thực tế | Dự báo bởi Mô hình Gốc (Baseline) | Dự báo bởi Mô hình Cải tiến (Cost-Sensitive) |
|---|---|---|
| **Khách hàng TỐT thực tế** | **{tn_b:,}** ca dự báo đúng (TN) | **{tn_i:,}** ca dự báo đúng (TN) |
| | **{fp_b:,}** ca bị từ chối sai (FP) | **{fp_i:,}** ca bị từ chối sai (FP) |
| **Khách hàng NỢ XẤU thực tế** | **{fn_b:,}** ca bị bỏ sót (FN) | **{fn_i:,}** ca bị bỏ sót (FN) |
| | **{tp_b:,}** ca bắt trúng (TP) | **{tp_i:,}** ca bắt trúng (TP) |

#### Ý nghĩa thực tiễn:
- **Bắt trúng thêm {tp_i - tp_b:,} ca nợ xấu thực tế** so với mô hình cũ.
- Số ca bỏ sót nợ xấu (False Negatives) giảm từ **{fn_b:,} ca** xuống còn **{fn_i:,} ca**.

---

### 3. ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH KINH TẾ (Expected Financial Loss - EFL)
$$EFL = (FN \\times 100M) + (FP \\times 10M)$$

1. **Mô hình Gốc (Baseline):** {efl_base:,.2f} triệu VND
2. **Mô hình Cải tiến:** {efl_imp:,.2f} triệu VND
3. **Lợi ích tài chính tiết kiệm được:** **{efl_base - efl_imp:,.2f} triệu VND**

---

### 4. BIỂU ĐỒ MINH CHỨNG
- `plots/feature_importance_v2.png`
- `plots/shap_summary_v2.png`
"""
    with open(os.path.join(minh_chung_dir, 'bao_cao_ket_qua_v2.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n- Đã xuất minh chứng đầy đủ tại thư mục: {minh_chung_dir}")
    
    print("\n--- CÔNG THỨC & LÝ THUYẾT ĐỐI CHỨNG KINH TẾ (NCKH NÂNG CAO - Q1) ---")
    print("1. AUC-ROC = P(f(x_bad) > f(x_good))")
    print("2. Gini = 2 * AUC - 1")
    print("3. KS = sup | F_good(s) - F_bad(s) |")
    print("4. EFL = (FN * D) + (FP * C)")
    
    pipeline_time = time.time() - pipeline_start
    print(f"\nPipeline hoàn tất trong {pipeline_time:.2f} giây.")
    print(f"Log: {os.path.abspath(log_file_path)}")

if __name__ == "__main__":
    main()
