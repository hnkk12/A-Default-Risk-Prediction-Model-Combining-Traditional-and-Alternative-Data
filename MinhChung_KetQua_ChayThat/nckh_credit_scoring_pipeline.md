# NGHIÊN CỨU HỆ THỐNG CHẤM ĐIỂM TÍN DỤNG & DỰ BÁO RỦI RO BÙNG NỢ
## Credit Scoring & Default Risk Prediction Pipeline (Q1 Journal Standard)

Notebook này kế thừa toàn bộ quy trình tiền xử lý dữ liệu hành vi thay thế, huấn luyện và so sánh mô hình chéo, kiểm định ý nghĩa thống kê, cải tiến mô hình bằng Cost-Sensitive Learning (phạt trọng số lỗi phân loại), giải thích AI bằng giá trị Shapley (SHAP) và tối ưu hóa lợi ích kinh tế cho Ngân hàng thương mại.

### Quy trình bao gồm:
1. **Tích hợp dữ liệu đa nguồn** (Traditional & Alternative Data)
2. **Thiết kế đặc trưng tài chính nâng cao** (Feature Engineering)
3. **Huấn luyện chéo (3-Fold Stratified Cross-Validation) & So sánh 4 thuật toán**: Random Forest, LightGBM, XGBoost, CatBoost
4. **Kiểm định ý nghĩa thống kê Paired t-test** chứng minh tính ưu việt của mô hình tốt nhất
5. **Cải tiến bằng Cost-Sensitive Learning** (giảm thiểu Expected Financial Loss - EFL)
6. **Giải thích mô hình cục bộ & toàn cục** bằng Feature Importance & SHAP Values.

### 1. Khai báo các thư viện cần thiết

```python
import pandas as pd
import numpy as np
import os
import time
import re
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
print("Đã import thành công các thư viện cần thiết.")
```

### 2. Tiền xử lý dữ liệu và tích hợp đa nguồn
Hệ thống sẽ tích hợp dữ liệu truyền thống (`application_train.csv`) với lịch sử tín dụng Bureau (`bureau.csv`), thông tin đơn vay cũ (`previous_application.csv`), và lịch sử thanh toán trễ hạn (`installments_payments.csv`).

```python
# Định nghĩa các hàm tiền xử lý dữ liệu gốc
def preprocess_and_merge_data(input_dir='../Input'):
    print("Đang nạp application_train.csv...")
    app_train = pd.read_csv(os.path.join(input_dir, 'application_train.csv'))
    
    print("Đang nạp và tổng hợp bureau.csv...")
    bureau = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), usecols=['SK_ID_CURR', 'DAYS_CREDIT', 'SK_ID_BUREAU'])
    bureau_agg = bureau.groupby('SK_ID_CURR', as_index=False).agg({
        'DAYS_CREDIT': 'mean',
        'SK_ID_BUREAU': 'count'
    })
    bureau_agg.columns = ['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    
    print("Đang nạp và tổng hợp previous_application.csv...")
    prev = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), usecols=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT', 'NAME_CONTRACT_STATUS'])
    prev['IS_REJECTED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    prev_agg = prev.groupby('SK_ID_CURR', as_index=False).agg({
        'SK_ID_PREV': 'count',
        'AMT_CREDIT': 'mean',
        'IS_REJECTED': 'sum'
    })
    prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
    
    print("Đang nạp và tổng hợp installments_payments.csv...")
    inst = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), usecols=['SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'])
    inst['PAY_DELAY'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
    inst['UNDERPAY'] = (inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']).clip(lower=0)
    inst_agg = inst.groupby('SK_ID_CURR', as_index=False).agg({
        'PAY_DELAY': 'mean',
        'UNDERPAY': 'mean'
    })
    inst_agg.columns = ['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
    
    print("Đang gộp các bảng dữ liệu lại với nhau...")
    df_combined = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df_combined = df_combined.merge(prev_agg, on='SK_ID_CURR', how='left')
    df_combined = df_combined.merge(inst_agg, on='SK_ID_CURR', how='left')
    
    return df_combined

# Đọc dữ liệu sạch đã lưu sẵn từ bước trước để tiết kiệm thời gian hoặc chạy tiền xử lý mới
cleaned_data_path = 'cleaned_data.csv'
if os.path.exists(cleaned_data_path):
    print("Tìm thấy cleaned_data.csv. Đang nạp...")
    df = pd.read_csv(cleaned_data_path)
else:
    print("Không tìm thấy cleaned_data.csv tại đây, đang thực hiện xử lý dữ liệu gốc từ thư mục ../Input...")
    df = preprocess_and_merge_data('../Input')
    # df.to_csv(cleaned_data_path, index=False)
print(f"Kích thước bộ dữ liệu: {df.shape[0]:,} dòng, {df.shape[1]:,} cột.")
```

### 3. Thiết kế đặc trưng tài chính nâng cao (Feature Engineering)
Ta tiến hành xây dựng các chỉ số kinh tế quan trọng để gia tăng khả năng dự báo rủi ro bùng nợ:
- **DIR (Debt-to-Income Ratio)**: Tỷ lệ Dư nợ trên Thu nhập.
- **AIR (Annuity-to-Income Ratio)**: Tỷ lệ Phải trả định kỳ trên Thu nhập.
- **ACR (Annuity-to-Credit Ratio)**: Tỷ lệ Phải trả định kỳ trên Tổng khoản vay.
- **DAR (Debt-to-Age Ratio)**: Tỷ trọng thâm niên làm việc trên tuổi đời.
- **EXT_SOURCES_PROD / EXT_SOURCES_MEAN**: Các tương tác tổ hợp từ nguồn điểm tín dụng bên thứ ba.

```python
# Xử lý dị biệt thâm niên
if 'DAYS_EMPLOYED' in df.columns:
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

print("Tạo các đặc trưng kinh tế học...")
df['DIR'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
df['AIR'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
df['ACR'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
    df['DAR'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)

print("Tạo các biến tương tác từ điểm tín dụng (Credit Bureau scores)...")
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
df_temp = df[ext_sources].fillna(df[ext_sources].median())
df['EXT_SOURCES_PROD'] = df_temp['EXT_SOURCE_1'] * df_temp['EXT_SOURCE_2'] * df_temp['EXT_SOURCE_3']
df['EXT_SOURCES_MEAN'] = df_temp[ext_sources].mean(axis=1)
df['EXT_SOURCE_2_3_MULT'] = df_temp['EXT_SOURCE_2'] * df_temp['EXT_SOURCE_3']

y = df['TARGET']
cols_to_drop = ['TARGET', 'SK_ID_CURR']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Mã hóa biến phân loại
le = LabelEncoder()
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if X[col].nunique() <= 2:
        X[col] = le.fit_transform(X[col].astype(str))

X = pd.get_dummies(X, drop_first=True)
X.columns = [re.sub(r'[\[\]\{\},:\s"\'\(\)]', '_', str(col)) for col in X.columns]
X = X.fillna(X.median())
print(f"Bộ dữ liệu huấn luyện cuối cùng: {X.shape[0]:,} mẫu và {X.shape[1]:,} đặc trưng.")
```

### 4. Huấn luyện chéo 3-Fold Stratified Cross-Validation & So sánh 4 thuật toán
Ta sử dụng cấu trúc phân tầng (`StratifiedKFold`) để đảm bảo tỷ lệ nợ xấu ở các fold đều nhau. 4 thuật toán mạnh nhất cho dữ liệu bảng sẽ được so sánh:
1. **Random Forest**
2. **LightGBM**
3. **XGBoost**
4. **CatBoost**

Các chỉ số đo lường:
- **AUC-ROC**: Khả năng phân biệt tốt/xấu.
- **Gini Coefficient**: $2 \times AUC - 1$.
- **Kolmogorov-Smirnov (KS)**: Độ phân tách phân phối tối đa giữa nhóm nợ tốt và nợ xấu.
- **Brier Score**: Sai số bình phương trung bình của xác suất dự báo (độ hiệu chuẩn).

```python
def calculate_ks(y_true, y_prob):
    df_ks = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df_ks = df_ks.sort_values(by='y_prob', ascending=False)
    total_goods = (df_ks['y_true'] == 0).sum()
    total_bads = (df_ks['y_true'] == 1).sum()
    if total_goods == 0 or total_bads == 0:
        return 0.0
    df_ks['cum_goods'] = (df_ks['y_true'] == 0).cumsum() / total_goods
    df_ks['cum_bads'] = (df_ks['y_true'] == 1).cumsum() / total_bads
    return float(max(abs(df_ks['cum_goods'] - df_ks['cum_bads'])))

models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12, n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=150, learning_rate=0.05, depth=5, verbose=0, thread_count=-1)
}

cv_results = {m_name: {"auc": [], "gini": [], "ks": [], "brier": []} for m_name in models.keys()}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Đang chạy Fold {fold + 1}/3 ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    for m_name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        
        auc = float(roc_auc_score(y_va, probs))
        gini = 2 * auc - 1
        ks = calculate_ks(y_va, probs)
        brier = float(brier_score_loss(y_va, probs))
        
        cv_results[m_name]["auc"].append(auc)
        cv_results[m_name]["gini"].append(gini)
        cv_results[m_name]["ks"].append(ks)
        cv_results[m_name]["brier"].append(brier)
        print(f"  * {m_name:<15} - AUC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f} | Brier: {brier:.4f} ({time.time()-t0:.2f}s)")
```

### 5. Bảng tổng hợp hiệu năng và Kiểm định Paired t-test

```python
print("="*80)
print("BẢNG TỔNG HỢP HIỆU NĂNG MÔ HÌNH (3-FOLD CROSS-VALIDATION)")
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
    print(f'{m_name:<15} | {auc_str:<22} | {gini_mean:<17.4f} | {ks_mean:<14.4f} | {brier_mean:<12.4f}')

best_model_name = max(summary_auc, key=summary_auc.get)
print("="*80)
print(f"Mô hình tối ưu nhất được lựa chọn: {best_model_name}")

print("\n--- Kiểm định ý nghĩa thống kê (Paired t-test) so với mô hình tốt nhất ---")
best_scores = cv_results[best_model_name]["auc"]
for m_name in models.keys():
    if m_name != best_model_name:
        stat, p_val = ttest_rel(best_scores, cv_results[m_name]["auc"])
        significance = "Có ý nghĩa (p < 0.05)" if p_val < 0.05 else "Không có ý nghĩa (p >= 0.05)"
        print(f"  - {best_model_name} vs {m_name:<15} -> p-value: {p_val:.5f} | Kết luận: {significance}")
```

### 6. Cải tiến mô hình bằng Cost-Sensitive Learning & Tính toán Tổn thất kinh tế kỳ vọng (EFL)
Do dữ liệu bị mất cân bằng trầm trọng (nhóm nợ xấu chỉ chiếm ~8%), nếu sử dụng hàm mất mát thông thường, mô hình sẽ thiên lệch về nhóm tốt, dẫn đến bỏ sót nhiều ca nợ xấu (False Negative cao).

Ta áp dụng hình phạt trọng số lớp lỗi `scale_pos_weight = 11.5` trên mô hình LightGBM tối ưu để tối ưu hóa quyết định tài chính.

#### Công thức Tổn thất Tài chính Kỳ vọng (EFL):
$$EFL = (FN \times D) + (FP \times C)$$
- $D$ (Tổn thất trốn nợ thực tế khi bỏ sót nợ xấu): Giả định trung bình $100$ triệu VND/dòng.
- $C$ (Chi phí cơ hội mất khách hàng tốt do từ chối nhầm): Giả định trung bình $10$ triệu VND/dòng.

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Huấn luyện mô hình Gốc (Không phạt)
final_base_model = lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
final_base_model.fit(X_train, y_train)
probs_base = final_base_model.predict_proba(X_val)[:, 1]
y_pred_base = final_base_model.predict(X_val)

# Huấn luyện mô hình Cải tiến (Cost-Sensitive, scale_pos_weight = 11.5)
improved_model = lgb.LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11.5, n_jobs=-1)
improved_model.fit(X_train, y_train)
probs_imp = improved_model.predict_proba(X_val)[:, 1]
y_pred_imp = improved_model.predict(X_val)

# Tính toán Confusion Matrix
cm_base = confusion_matrix(y_val, y_pred_base)
cm_imp = confusion_matrix(y_val, y_pred_imp)
tn_base, fp_base, fn_base, tp_base = map(int, cm_base.ravel())
tn_imp, fp_imp, fn_imp, tp_imp = map(int, cm_imp.ravel())
diff_tp = tp_imp - tp_base

# Tính toán Tổn thất tài chính kỳ vọng (EFL)
D_loss = 100.0  # triệu VND/ca
C_opp = 10.0    # triệu VND/ca
efl_base = (fn_base * D_loss) + (fp_base * C_opp)
efl_imp = (fn_imp * D_loss) + (fp_imp * C_opp)
savings = efl_base - efl_imp

print("-"*70)
print(f"{'Chỉ Số Đánh Giá':<40} | {'Mô Hình Gốc':<13} | {'Mô Hình Cải Tiến'}")
print("-"*70)
print(f"{'AUC-ROC Score':<40} | {roc_auc_score(y_val, probs_base):<13.4f} | {roc_auc_score(y_val, probs_imp):.4f}")
print(f"{'Hệ Số Gini':<40} | {(2*roc_auc_score(y_val, probs_base)-1):<13.4f} | {(2*roc_auc_score(y_val, probs_imp)-1):.4f}")
print(f"{'Kolmogorov-Smirnov (KS)':<40} | {calculate_ks(y_val, probs_base):<13.4f} | {calculate_ks(y_val, probs_imp):.4f}")
print(f"{'Brier Score (Độ hiệu chuẩn xác suất)':<40} | {brier_score_loss(y_val, probs_base):<13.4f} | {brier_score_loss(y_val, probs_imp):.4f}")
print(f"{'Khách Hàng TỐT Dự Báo Đúng (TN)':<40} | {tn_base:<13} | {tn_imp}")
print(f"{'Khách Hàng TỐT Dự Báo Sai Thành Xấu (FP)':<40} | {fp_base:<13} | {fp_imp}")
print(f"{'BỎ SÓT Nợ Xấu Thực Tế (FN - Lỗi Loại II)':<40} | {fn_base:<13} | {fn_imp}")
print(f"{'BẮT TRÚNG Nợ Xấu Thực Tế (TP - Lỗi Loại I)':<40} | {tp_base:<13} | {tp_imp}")
print(f"{'Expected Financial Loss (EFL)':<40} | {efl_base:<13,.2f} | {efl_imp:,.2f} triệu VND")
print("-"*70)
print(f"=> Nhận xét: Mô hình Cost-Sensitive bắt trúng thêm {diff_tp:,} ca nợ xấu thực tế.")
print(f"=> Tiết kiệm được cho Ngân hàng: {savings:,.2f} triệu VND.")
```

### 7. Trực quan hóa Đặc trưng quan trọng & Giải thích AI bằng SHAP (XAI)
Chúng ta sẽ vẽ biểu đồ độ quan trọng đặc trưng toàn cục và sử dụng SHAP values để giải thích tính tác động phi tuyến của các đặc trưng đến rủi ro bùng nợ.

```python
# 7.1. Vẽ Feature Importance
importances = improved_model.feature_importances_ / improved_model.feature_importances_.sum()
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
top_15 = feat_imp.head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_15, palette='viridis')
plt.title('Top 15 Feature Importances (Improved LightGBM)')
plt.xlabel('Normalized Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 7.2. Tính toán SHAP Values trên mẫu đại diện 500 khách hàng
print("Đang chạy SHAP Explainer (có thể mất khoảng 10-20 giây)...")
explainer = shap.TreeExplainer(improved_model)
X_val_sample = X_val.sample(n=500, random_state=42)
shap_values = explainer(X_val_sample)

plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_val_sample, show=False)
plt.title("SHAP Summary Plot - Improved LightGBM", fontsize=14, pad=15)
plt.tight_layout()
plt.show()
```