import pandas as pd
import numpy as np
import os
import time

def main():
    start_time = time.time()
    
    input_dir = 'Input2'
    output_dir = 'data_output_v2'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Đang xử lý dữ liệu Việt hóa (Version 2)... Vui lòng đợi.")
    
    # 1. Load dữ liệu hồ sơ chính (ho_so_khach_hang_train.csv)
    app_train_path = os.path.join(input_dir, 'ho_so_khach_hang_train.csv')
    if not os.path.exists(app_train_path):
        print(f"Lỗi: Không tìm thấy file {app_train_path}")
        return
    print("- Đang nạp ho_so_khach_hang_train.csv...")
    app_train = pd.read_csv(app_train_path)
    
    # 2. Load và xử lý dữ liệu thay thế Bureau (lich_su_tin_dung_bureau.csv)
    bureau_path = os.path.join(input_dir, 'lich_su_tin_dung_bureau.csv')
    if not os.path.exists(bureau_path):
        print(f"Lỗi: Không tìm thấy file {bureau_path}")
        return
    print("- Đang nạp và tổng hợp lich_su_tin_dung_bureau.csv...")
    bureau = pd.read_csv(bureau_path, usecols=['ma_khach_hang', 'so_ngay_truoc_khi_vay_ngoai', 'ma_khoan_vay_ngoai'])
    bureau_agg = bureau.groupby('ma_khach_hang', as_index=False).agg({
        'so_ngay_truoc_khi_vay_ngoai': 'mean',
        'ma_khoan_vay_ngoai': 'count'
    })
    bureau_agg.columns = ['ma_khach_hang', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    
    # 3. Load và xử lý dữ liệu hồ sơ vay cũ (don_vay_cu_noi_bo.csv)
    prev_path = os.path.join(input_dir, 'don_vay_cu_noi_bo.csv')
    if os.path.exists(prev_path):
        print("- Đang nạp và tổng hợp don_vay_cu_noi_bo.csv...")
        prev = pd.read_csv(prev_path, usecols=['ma_khach_hang', 'ma_don_vay_cu', 'so_tien_duoc_duyet_vnd', 'ket_qua_duyet'])
        prev['IS_REJECTED'] = (prev['ket_qua_duyet'] == 'Tu choi').astype(int)
        prev_agg = prev.groupby('ma_khach_hang', as_index=False).agg({
            'ma_don_vay_cu': 'count',
            'so_tien_duoc_duyet_vnd': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['ma_khach_hang', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
    else:
        print("- Cảnh báo: Không tìm thấy don_vay_cu_noi_bo.csv, bỏ qua đặc trưng này.")
        prev_agg = pd.DataFrame(columns=['ma_khach_hang', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT'])
        
    # 4. Load và xử lý lịch sử đóng tiền (lich_su_tra_no_dinh_ky.csv)
    inst_path = os.path.join(input_dir, 'lich_su_tra_no_dinh_ky.csv')
    if os.path.exists(inst_path):
        print("- Đang nạp và tổng hợp lich_su_tra_no_dinh_ky.csv...")
        inst = pd.read_csv(inst_path, usecols=['ma_khach_hang', 'ngay_phai_tra', 'ngay_thuc_te_tra', 'so_tien_phai_tra_vnd', 'so_tien_thuc_te_tra_vnd'])
        # Tính số ngày đóng trễ hạn (nếu ngay_thuc_te_tra > ngay_phai_tra)
        inst['PAY_DELAY'] = (inst['ngay_thuc_te_tra'] - inst['ngay_phai_tra']).clip(lower=0)
        # Tính số tiền còn nợ/thiếu của đợt đóng tiền
        inst['UNDERPAY'] = (inst['so_tien_phai_tra_vnd'] - inst['so_tien_thuc_te_tra_vnd']).clip(lower=0)
        
        inst_agg = inst.groupby('ma_khach_hang', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['ma_khach_hang', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
    else:
        print("- Cảnh báo: Không tìm thấy lich_su_tra_no_dinh_ky.csv, bỏ qua đặc trưng này.")
        inst_agg = pd.DataFrame(columns=['ma_khach_hang', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN'])
        
    # 5. Gộp tất cả các bảng dữ liệu lại với nhau (Merge)
    print("- Đang gộp các bảng dữ liệu...")
    df_combined = app_train.merge(bureau_agg, on='ma_khach_hang', how='left')
    
    if not prev_agg.empty:
        df_combined = df_combined.merge(prev_agg, on='ma_khach_hang', how='left')
    if not inst_agg.empty:
        df_combined = df_combined.merge(inst_agg, on='ma_khach_hang', how='left')
        
    num_rows, num_cols = df_combined.shape
    
    # Tính tỷ lệ mất cân bằng TARGET
    target_counts = df_combined['TARGET'].value_counts(normalize=True) * 100
    percent_0 = target_counts.get(0, 0)
    percent_1 = target_counts.get(1, 0)
    
    # Lưu file đã làm sạch
    output_path = os.path.join(output_dir, 'cleaned_data2.csv')
    print(f"- Đang xuất dữ liệu ra file '{output_path}'...")
    df_combined.to_csv(output_path, index=False)
    
    execution_duration = time.time() - start_time
    
    # In kết quả
    print("\n" + "="*60)
    print(f"Kích thước dữ liệu: {num_rows:,} hàng và {num_cols} biến.")
    print(f"Tỷ lệ TARGET: Nhóm tốt (0) chiếm {percent_0:.2f}%, Nhóm nợ xấu (1) chiếm {percent_1:.2f}%.")
    print(f"Xuất file cleaned_data2.csv hoàn tất trong {execution_duration:.2f} giây.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
