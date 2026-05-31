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
    print("  HỆ THỐNG PHÊ DUYỆT TÍN DỤNG TỰ ĐỘNG V2 (DỰ BÁO HỒ SƠ KHÁCH HÀNG MỚI VIỆT HÓA)")
    print("=================================================================================")
    
    # 1. Nạp dữ liệu huấn luyện để dạy mô hình
    train_file = os.path.join('data_output_v2', 'cleaned_data2.csv')
    if not os.path.exists(train_file):
        print("Lỗi: Không tìm thấy tệp dữ liệu đã làm sạch tại 'data_output_v2/cleaned_data2.csv'.")
        print("Vui lòng chạy file step1_2_data_processing_v2.py trước.")
        return
        
    print("1. Đang nạp dữ liệu quá khứ để huấn luyện bộ não mô hình...")
    df_train = pd.read_csv(train_file)
    
    # Xử lý đặc trưng tập huấn luyện
    if 'so_ngay_lam_viec' in df_train.columns:
        df_train['so_ngay_lam_viec_ANOM'] = df_train["so_ngay_lam_viec"] == 365243
        df_train['so_ngay_lam_viec'] = df_train['so_ngay_lam_viec'].replace({365243: np.nan})
        
    df_train['DIR'] = df_train['so_tien_vay_vnd'] / (df_train['thu_nhap_nam_vnd'] + 1e-5)
    df_train['AIR'] = df_train['khoan_tra_dinh_ky_vnd'] / (df_train['thu_nhap_nam_vnd'] / 12 + 1e-5)
    df_train['ACR'] = df_train['khoan_tra_dinh_ky_vnd'] / (df_train['so_tien_vay_vnd'] + 1e-5)
    if 'so_ngay_lam_viec' in df_train.columns and 'tuoi_doi_ngay' in df_train.columns:
        df_train['DAR'] = df_train['so_ngay_lam_viec'] / (df_train['tuoi_doi_ngay'] + 1e-5)
        
    ext_sources = ['diem_tin_dung_nguon_1', 'diem_tin_dung_nguon_2', 'diem_tin_dung_nguon_3']
    df_temp_train = df_train[ext_sources].fillna(df_train[ext_sources].median())
    df_train['EXT_SOURCES_PROD'] = df_temp_train['diem_tin_dung_nguon_1'] * df_temp_train['diem_tin_dung_nguon_2'] * df_temp_train['diem_tin_dung_nguon_3']
    df_train['EXT_SOURCES_MEAN'] = df_temp_train[ext_sources].mean(axis=1)
    df_train['EXT_SOURCE_2_3_MULT'] = df_temp_train['diem_tin_dung_nguon_2'] * df_temp_train['diem_tin_dung_nguon_3']
    
    y_train = df_train['TARGET']
    X_train = df_train.drop(columns=['TARGET', 'ma_khach_hang', 'ho_va_ten'], errors='ignore')
    
    # Lưu vết các LabelEncoder của tập Train để map chuẩn sang tập Test
    label_encoders = {}
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X_train[col].nunique() <= 2:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            label_encoders[col] = le
            
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
    
    # 2. Đọc khách hàng mới
    print("\n" + "-"*50)
    print("CẤU HÌNH DỮ LIỆU KHÁCH HÀNG MỚI CẦN DỰ BÁO V2")
    print("-"*50)
    
    default_test_path = os.path.join('Input2', 'ho_so_khach_hang_test.csv')
    custom_path = input("Nhập đường dẫn file CSV chứa khách hàng mới\n(Ấn Enter để dùng mặc định 'Input2/ho_so_khach_hang_test.csv'): ").strip()
    
    if not custom_path:
        test_path = default_test_path
        num_rows_input = input("Nhập số lượng khách hàng muốn dự báo (Ấn Enter để dùng mặc định 20): ").strip()
        num_rows = int(num_rows_input) if num_rows_input.isdigit() else 20
    else:
        test_path = custom_path
        num_rows = None
        
    if not os.path.exists(test_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{test_path}'")
        return
        
    print(f"\n3. Đang trích xuất thông tin khách hàng mới từ '{test_path}'...")
    if num_rows:
        df_new = pd.read_csv(test_path, nrows=num_rows)
    else:
        df_new = pd.read_csv(test_path)
        
    new_ids = df_new['ma_khach_hang'].tolist()
    
    # A. Tích hợp dữ liệu lịch sử Bureau tín dụng
    bureau_path = os.path.join('Input2', 'lich_su_tin_dung_bureau.csv')
    if os.path.exists(bureau_path):
        bureau = pd.read_csv(bureau_path)
        bureau_subset = bureau[bureau['ma_khach_hang'].isin(new_ids)]
        bureau_agg = bureau_subset.groupby('ma_khach_hang', as_index=False).agg({
            'so_ngay_truoc_khi_vay_ngoai': 'mean',
            'ma_khoan_vay_ngoai': 'count'
        })
        bureau_agg.columns = ['ma_khach_hang', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
        df_new = df_new.merge(bureau_agg, on='ma_khach_hang', how='left')
    else:
        df_new['BURO_DAYS_CREDIT_MEAN'] = np.nan
        df_new['BURO_COUNT'] = np.nan
        
    # B. Tích hợp dữ liệu hồ sơ vay cũ
    prev_path = os.path.join('Input2', 'don_vay_cu_noi_bo.csv')
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        prev_subset = prev[prev['ma_khach_hang'].isin(new_ids)]
        prev_subset['IS_REJECTED'] = (prev_subset['ket_qua_duyet'] == 'Tu choi').astype(int)
        prev_agg = prev_subset.groupby('ma_khach_hang', as_index=False).agg({
            'ma_don_vay_cu': 'count',
            'so_tien_duoc_duyet_vnd': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['ma_khach_hang', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
        df_new = df_new.merge(prev_agg, on='ma_khach_hang', how='left')
    else:
        df_new['PREV_APP_COUNT'] = np.nan
        df_new['PREV_APP_CREDIT_MEAN'] = np.nan
        df_new['PREV_APP_REJECTED_COUNT'] = np.nan
        
    # C. Tích hợp lịch sử đóng tiền của khoản vay cũ
    inst_path = os.path.join('Input2', 'lich_su_tra_no_dinh_ky.csv')
    if os.path.exists(inst_path):
        inst = pd.read_csv(inst_path)
        inst_subset = inst[inst['ma_khach_hang'].isin(new_ids)]
        inst_subset['PAY_DELAY'] = (inst_subset['ngay_thuc_te_tra'] - inst_subset['ngay_phai_tra']).clip(lower=0)
        inst_subset['UNDERPAY'] = (inst_subset['so_tien_phai_tra_vnd'] - inst_subset['so_tien_thuc_te_tra_vnd']).clip(lower=0)
        inst_agg = inst_subset.groupby('ma_khach_hang', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['ma_khach_hang', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
        df_new = df_new.merge(inst_agg, on='ma_khach_hang', how='left')
    else:
        df_new['INST_PAY_DELAY_MEAN'] = np.nan
        df_new['INST_UNDERPAY_MEAN'] = np.nan
        
    # Kiểm tra bổ sung cột bắt buộc nếu thiếu
    required_cols = {
        'tuoi_doi_ngay': -14600,
        'thu_nhap_nam_vnd': 120000000,
        'so_tien_vay_vnd': 300000000,
        'khoan_tra_dinh_ky_vnd': 15000000
    }
    for r_col, default_val in required_cols.items():
        if r_col not in df_new.columns:
            df_new[r_col] = default_val
            
    original_display = df_new[['ma_khach_hang', 'ho_va_ten', 'tuoi_doi_ngay', 'thu_nhap_nam_vnd', 'so_tien_vay_vnd']].copy()
    original_display['Age'] = (original_display['tuoi_doi_ngay'] / -365).fillna(40).astype(int)
    
    print("4. Đang tiến hành tiền xử lý và thiết kế đặc trưng cho khách hàng mới...")
    if 'so_ngay_lam_viec' in df_new.columns:
        df_new['so_ngay_lam_viec_ANOM'] = df_new["so_ngay_lam_viec"] == 365243
        df_new['so_ngay_lam_viec'] = df_new['so_ngay_lam_viec'].replace({365243: np.nan})
        
    df_new['DIR'] = df_new['so_tien_vay_vnd'] / (df_new['thu_nhap_nam_vnd'] + 1e-5)
    df_new['AIR'] = df_new['khoan_tra_dinh_ky_vnd'] / (df_new['thu_nhap_nam_vnd'] / 12 + 1e-5)
    df_new['ACR'] = df_new['khoan_tra_dinh_ky_vnd'] / (df_new['so_tien_vay_vnd'] + 1e-5)
    if 'so_ngay_lam_viec' in df_new.columns and 'tuoi_doi_ngay' in df_new.columns:
        df_new['DAR'] = df_new['so_ngay_lam_viec'] / (df_new['tuoi_doi_ngay'] + 1e-5)
        
    df_temp_new = df_new[ext_sources].copy()
    for col in ext_sources:
        if col not in df_temp_new.columns:
            df_temp_new[col] = np.nan
    df_temp_new = df_temp_new.fillna(df_train[ext_sources].median())
    
    df_new['EXT_SOURCES_PROD'] = df_temp_new['diem_tin_dung_nguon_1'] * df_temp_new['diem_tin_dung_nguon_2'] * df_temp_new['diem_tin_dung_nguon_3']
    df_new['EXT_SOURCES_MEAN'] = df_temp_new[ext_sources].mean(axis=1)
    df_new['EXT_SOURCE_2_3_MULT'] = df_temp_new['diem_tin_dung_nguon_2'] * df_temp_new['diem_tin_dung_nguon_3']
    
    X_new = df_new.drop(columns=['TARGET', 'ma_khach_hang', 'ho_va_ten'], errors='ignore')
    
    # Ánh xạ LabelEncoder từ tập huấn luyện gốc (Tránh lệch nhãn khi tập test quá nhỏ)
    for col in categorical_cols:
        if col in X_new.columns and col in label_encoders:
            le = label_encoders[col]
            X_new[col] = X_new[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            
    X_new = pd.get_dummies(X_new, drop_first=True)
    X_new.columns = [re.sub(r'[\[\]\{\},:\s"\'\(\)]', '_', str(col)) for col in X_new.columns]
    
    for col in X_train.columns:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[X_train.columns]
    X_new = X_new.fillna(train_medians)
    
    print("5. Đang chạy mô hình dự báo rủi ro tín dụng...")
    probs = model.predict_proba(X_new)[:, 1]
    original_display['Prob_Default'] = probs * 100
    
    # LƯU Ý LÝ THUYẾT: Vì mô hình dùng scale_pos_weight = 11.5 (Cost-Sensitive),
    # Xác suất dự báo bị kéo lệch (pushed) lên cao để tránh bỏ sót nợ xấu.
    # Một ngưỡng rủi ro thực tế 15.0% tương đương với ngưỡng mô hình cải tiến là ~65.0%.
    # Cấu hình ngưỡng phê duyệt thực tế đã được chuẩn hóa lại ở mức 65%.
    threshold = 65.0
    original_display['Decision'] = original_display['Prob_Default'].apply(
        lambda p: '❌ TỪ CHỐI' if p >= threshold else '✅ PHÊ DUYỆT'
    )
    
    # 5. Hiển thị bảng kết quả trực tiếp ra màn hình
    print("\n" + "="*120)
    print(f"BẢNG KẾT QUẢ ĐÁNH GIÁ & QUYẾT ĐỊNH CHO VAY V2 (Ngưỡng rủi ro mô hình cải tiến: {threshold:.2f}%)")
    print("="*120)
    print(f"{'Mã KH':<8} | {'Họ và Tên':<22} | {'Tuổi':<4} | {'Thu Nhập Tháng':<16} | {'Khoản Vay (VND)':<16} | {'Xác suất nợ xấu':<16} | {'Quyết định'}")
    print("-"*120)
    for idx, row in original_display.iterrows():
        income_m = row['thu_nhap_nam_vnd'] / 12
        print(f"{int(row['ma_khach_hang']):<8} | {row['ho_va_ten']:<22} | {int(row['Age']):<4} | {income_m:<16,.0f} | {row['so_tien_vay_vnd']:<16,.0f} | {row['Prob_Default']:<15.2f}% | {row['Decision']}")
    print("="*120)
    
    # Lưu báo cáo vào thư mục results2
    results_dir = 'results2'
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"new_customers_eval_v2_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=================================================================================\n")
        f.write("   KẾT QUẢ ĐÁNH GIÁ RỦI RO & PHÊ DUYỆT TỰ ĐỘNG CHO KHÁCH HÀNG MỚI (VIỆT HÓA)\n")
        f.write("=================================================================================\n")
        f.write(f"Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tệp dữ liệu đầu vào: {test_path}\n")
        f.write(f"Ngưỡng rủi ro bùng nợ tối đa chấp nhận: {threshold}%\n\n")
        f.write("="*120 + "\n")
        f.write(f"{'Mã KH':<8} | {'Họ và Tên':<22} | {'Tuổi':<4} | {'Thu Nhập Tháng':<16} | {'Khoản Vay (VND)':<16} | {'Xác suất nợ xấu':<16} | {'Quyết định'}\n")
        f.write("-"*120 + "\n")
        for idx, row in original_display.iterrows():
            income_m = row['thu_nhap_nam_vnd'] / 12
            f.write(f"{int(row['ma_khach_hang']):<8} | {row['ho_va_ten']:<22} | {int(row['Age']):<4} | {income_m:<16,.0f} | {row['so_tien_vay_vnd']:<16,.0f} | {row['Prob_Default']:<15.2f}% | {row['Decision']}\n")
        f.write("="*120 + "\n")
        
    print(f"\nBáo cáo chi tiết đã được lưu trữ thành công tại: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()
