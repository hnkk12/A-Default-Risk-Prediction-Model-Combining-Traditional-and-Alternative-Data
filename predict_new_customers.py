import pandas as pd
import numpy as np
import os
import re
import time
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=================================================================================")
    print("       HỆ THỐNG PHÊ DUYỆT TÍN DỤNG TỰ ĐỘNG (DỰ BÁO HỒ SƠ KHÁCH HÀNG MỚI)")
    print("=================================================================================")
    
    # 1. Nạp dữ liệu huấn luyện để dạy mô hình
    train_file = os.path.join('data_output', 'cleaned_data.csv')
    if not os.path.exists(train_file):
        print("Lỗi: Không tìm thấy tệp dữ liệu đã làm sạch tại 'data_output/cleaned_data.csv'.")
        print("Vui lòng chạy file step1_2_data_processing.py trước.")
        return
        
    print("1. Đang nạp dữ liệu quá khứ để huấn luyện bộ não mô hình...")
    df_train = pd.read_csv(train_file)
    
    # Xử lý đặc trưng tập huấn luyện
    if 'DAYS_EMPLOYED' in df_train.columns:
        df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243
        df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].replace({365243: np.nan})
        
    df_train['DIR'] = df_train['AMT_CREDIT'] / (df_train['AMT_INCOME_TOTAL'] + 1e-5)
    df_train['AIR'] = df_train['AMT_ANNUITY'] / (df_train['AMT_INCOME_TOTAL'] + 1e-5)
    df_train['ACR'] = df_train['AMT_ANNUITY'] / (df_train['AMT_CREDIT'] + 1e-5)
    if 'DAYS_EMPLOYED' in df_train.columns and 'DAYS_BIRTH' in df_train.columns:
        df_train['DAR'] = df_train['DAYS_EMPLOYED'] / (df_train['DAYS_BIRTH'] + 1e-5)
        
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df_temp_train = df_train[ext_sources].fillna(df_train[ext_sources].median())
    df_train['EXT_SOURCES_PROD'] = df_temp_train['EXT_SOURCE_1'] * df_temp_train['EXT_SOURCE_2'] * df_temp_train['EXT_SOURCE_3']
    df_train['EXT_SOURCES_MEAN'] = df_temp_train[ext_sources].mean(axis=1)
    df_train['EXT_SOURCE_2_3_MULT'] = df_temp_train['EXT_SOURCE_2'] * df_temp_train['EXT_SOURCE_3']
    
    y_train = df_train['TARGET']
    X_train = df_train.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # Mã hóa nhãn dạng phân loại
    le = LabelEncoder()
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X_train[col].nunique() <= 2:
            X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_train.columns = [re.sub(r'[\[\]\{\},:\s"\'\(\)]', '_', str(col)) for col in X_train.columns]
    
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    
    print("2. Đang huấn luyện mô hình LightGBM tối ưu...")
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=11.5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   -> Huấn luyện mô hình thành công!")
    
    # 2. Đọc khách hàng mới thông qua dấu nhắc tương tác (Interactive prompt)
    print("\n" + "-"*50)
    print("CẤU HÌNH DỮ LIỆU KHÁCH HÀNG MỚI CẦN DỰ BÁO")
    print("-"*50)
    
    default_test_path = os.path.join('Input', 'application_test.csv')
    print("Mẹo: Bạn có thể thử nghiệm bằng cách nhập: new_customers/template.csv")
    custom_path = input("Nhập đường dẫn file CSV chứa khách hàng mới\n(Ấn Enter để dùng mặc định 'Input/application_test.csv'): ").strip()
    
    if not custom_path:
        test_path = default_test_path
        num_rows_input = input("Nhập số lượng khách hàng muốn dự báo (Ấn Enter để dùng mặc định 20): ").strip()
        num_rows = int(num_rows_input) if num_rows_input.isdigit() else 20
    else:
        test_path = custom_path
        num_rows = None # Đọc toàn bộ file khách hàng tùy chỉnh của bạn
        
    if not os.path.exists(test_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{test_path}'")
        return
        
    print(f"\n3. Đang trích xuất thông tin khách hàng mới từ '{test_path}'...")
    if num_rows:
        df_new = pd.read_csv(test_path, nrows=num_rows)
    else:
        df_new = pd.read_csv(test_path)
        
    new_ids = df_new['SK_ID_CURR'].tolist()
    
    # A. Tích hợp dữ liệu lịch sử Bureau tín dụng
    bureau_path = os.path.join('Input', 'bureau.csv')
    if os.path.exists(bureau_path):
        bureau = pd.read_csv(bureau_path)
        bureau_subset = bureau[bureau['SK_ID_CURR'].isin(new_ids)]
        bureau_agg = bureau_subset.groupby('SK_ID_CURR', as_index=False).agg({
            'DAYS_CREDIT': 'mean',
            'SK_ID_BUREAU': 'count'
        })
        bureau_agg.columns = ['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
        df_new = df_new.merge(bureau_agg, on='SK_ID_CURR', how='left')
    else:
        df_new['BURO_DAYS_CREDIT_MEAN'] = np.nan
        df_new['BURO_COUNT'] = np.nan
        
    # B. Tích hợp dữ liệu hồ sơ vay cũ
    prev_path = os.path.join('Input', 'previous_application.csv')
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        prev_subset = prev[prev['SK_ID_CURR'].isin(new_ids)]
        prev_subset['IS_REJECTED'] = (prev_subset['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
        prev_agg = prev_subset.groupby('SK_ID_CURR', as_index=False).agg({
            'SK_ID_PREV': 'count',
            'AMT_CREDIT': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
        df_new = df_new.merge(prev_agg, on='SK_ID_CURR', how='left')
    else:
        df_new['PREV_APP_COUNT'] = np.nan
        df_new['PREV_APP_CREDIT_MEAN'] = np.nan
        df_new['PREV_APP_REJECTED_COUNT'] = np.nan
        
    # C. Tích hợp lịch sử đóng tiền của khoản vay cũ
    inst_path = os.path.join('Input', 'installments_payments.csv')
    if os.path.exists(inst_path):
        inst = pd.read_csv(inst_path)
        inst_subset = inst[inst['SK_ID_CURR'].isin(new_ids)]
        inst_subset['PAY_DELAY'] = (inst_subset['DAYS_ENTRY_PAYMENT'] - inst_subset['DAYS_INSTALMENT']).clip(lower=0)
        inst_subset['UNDERPAY'] = (inst_subset['AMT_INSTALMENT'] - inst_subset['AMT_PAYMENT']).clip(lower=0)
        inst_agg = inst_subset.groupby('SK_ID_CURR', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
        df_new = df_new.merge(inst_agg, on='SK_ID_CURR', how='left')
    else:
        df_new['INST_PAY_DELAY_MEAN'] = np.nan
        df_new['INST_UNDERPAY_MEAN'] = np.nan
        
    # Kiểm tra và bổ sung các trường thông tin cơ bản nếu file tùy chỉnh bị thiếu
    required_cols = {
        'DAYS_BIRTH': -14600,       # mặc định 40 tuổi
        'AMT_INCOME_TOTAL': 120000,  # mặc định thu nhập trung bình
        'AMT_CREDIT': 300000,        # mặc định khoản vay trung bình
        'AMT_ANNUITY': 15000         # mặc định phí định kỳ trung bình
    }
    for r_col, default_val in required_cols.items():
        if r_col not in df_new.columns:
            df_new[r_col] = default_val
            
    # Lưu thông tin gốc để hiển thị báo cáo
    original_display = df_new[['SK_ID_CURR', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT']].copy()
    original_display['Age'] = (original_display['DAYS_BIRTH'] / -365).fillna(40).astype(int)
    
    # 3. Tiền xử lý dữ liệu khách hàng mới đồng bộ với dữ liệu huấn luyện
    print("4. Đang tiến hành tiền xử lý và thiết kế đặc trưng cho khách hàng mới...")
    if 'DAYS_EMPLOYED' in df_new.columns:
        df_new['DAYS_EMPLOYED_ANOM'] = df_new["DAYS_EMPLOYED"] == 365243
        df_new['DAYS_EMPLOYED'] = df_new['DAYS_EMPLOYED'].replace({365243: np.nan})
        
    df_new['DIR'] = df_new['AMT_CREDIT'] / (df_new['AMT_INCOME_TOTAL'] + 1e-5)
    df_new['AIR'] = df_new['AMT_ANNUITY'] / (df_new['AMT_INCOME_TOTAL'] + 1e-5)
    df_new['ACR'] = df_new['AMT_ANNUITY'] / (df_new['AMT_CREDIT'] + 1e-5)
    if 'DAYS_EMPLOYED' in df_new.columns and 'DAYS_BIRTH' in df_new.columns:
        df_new['DAR'] = df_new['DAYS_EMPLOYED'] / (df_new['DAYS_BIRTH'] + 1e-5)
        
    df_temp_new = df_new[ext_sources].copy()
    for col in ext_sources:
        if col not in df_temp_new.columns:
            df_temp_new[col] = np.nan
    df_temp_new = df_temp_new.fillna(df_train[ext_sources].median())
    
    df_new['EXT_SOURCES_PROD'] = df_temp_new['EXT_SOURCE_1'] * df_temp_new['EXT_SOURCE_2'] * df_temp_new['EXT_SOURCE_3']
    df_new['EXT_SOURCES_MEAN'] = df_temp_new[ext_sources].mean(axis=1)
    df_new['EXT_SOURCE_2_3_MULT'] = df_temp_new['EXT_SOURCE_2'] * df_temp_new['EXT_SOURCE_3']
    
    X_new = df_new.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    
    # Mã hóa phân loại
    for col in categorical_cols:
        if col in X_new.columns:
            if X_new[col].nunique() <= 2:
                X_new[col] = le.fit_transform(X_new[col].astype(str))
    X_new = pd.get_dummies(X_new, drop_first=True)
    X_new.columns = [re.sub(r'[\[\]\{\},:\s"\'\(\)]', '_', str(col)) for col in X_new.columns]
    
    # Căn chỉnh các cột của dữ liệu mới khớp chính xác với tập huấn luyện
    for col in X_train.columns:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[X_train.columns]
    
    # Điền giá trị thiếu bằng giá trị trung vị của tập huấn luyện gốc
    X_new = X_new.fillna(train_medians)
    
    # 4. Dự báo xác suất bùng nợ (Inference)
    print("5. Đang chạy mô hình dự báo rủi ro tín dụng...")
    probs = model.predict_proba(X_new)[:, 1]
    original_display['Prob_Default'] = probs * 100
    
    # Ngưỡng phê duyệt cấp tín dụng của Ngân hàng (Ví dụ thiết lập là 15%)
    threshold = 15.0
    original_display['Decision'] = original_display['Prob_Default'].apply(
        lambda p: '❌ TỪ CHỐI (Reject)' if p >= threshold else '✅ PHÊ DUYỆT (Approve)'
    )
    
    # 5. Hiển thị bảng kết quả trực tiếp ra màn hình
    print("\n" + "="*100)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ & QUYẾT ĐỊNH CHO VAY (Ngưỡng rủi ro tối đa chấp nhận: 15.00%)")
    print("="*100)
    print(f"{'Mã KH (ID)':<10} | {'Tuổi':<5} | {'Thu Nhập (USD)':<15} | {'Khoản Vay (USD)':<16} | {'Xác suất bùng nợ':<18} | {'Quyết định duyệt vay'}")
    print("-"*100)
    for idx, row in original_display.iterrows():
        print(f"{int(row['SK_ID_CURR']):<10} | {int(row['Age']):<5} | {row['AMT_INCOME_TOTAL']:<15,.2f} | {row['AMT_CREDIT']:<16,.2f} | {row['Prob_Default']:<17.2f}% | {row['Decision']}")
    print("="*100)
    
    # Lưu báo cáo vào thư mục results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"new_customers_eval_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=================================================================================\n")
        f.write("      KẾT QUẢ ĐÁNH GIÁ RỦI RO & PHÊ DUYỆT TỰ ĐỘNG CHO KHÁCH HÀNG MỚI\n")
        f.write("=================================================================================\n")
        f.write(f"Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tệp dữ liệu đầu vào: {test_path}\n")
        f.write(f"Ngưỡng rủi ro bùng nợ tối đa chấp nhận (Cut-off Threshold): {threshold}%\n\n")
        f.write("="*100 + "\n")
        f.write(f"{'Mã KH (ID)':<10} | {'Tuổi':<5} | {'Thu Nhập (USD)':<15} | {'Khoản Vay (USD)':<16} | {'Xác suất bùng nợ':<18} | {'Quyết định duyệt vay'}\n")
        f.write("-"*100 + "\n")
        for idx, row in original_display.iterrows():
            f.write(f"{int(row['SK_ID_CURR']):<10} | {int(row['Age']):<5} | {row['AMT_INCOME_TOTAL']:<15,.2f} | {row['AMT_CREDIT']:<16,.2f} | {row['Prob_Default']:<17.2f}% | {row['Decision']}\n")
        f.write("="*100 + "\n")
        
    print(f"\nBáo cáo chi tiết đã được lưu trữ thành công tại: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
