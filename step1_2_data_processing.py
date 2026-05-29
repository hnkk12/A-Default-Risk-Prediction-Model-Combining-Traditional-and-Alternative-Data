import pandas as pd
import numpy as np
import os
import time

def main():
    start_time = time.time()
    
    input_dir = 'Input'
    output_dir = 'data_output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Đang xử lý dữ liệu... Vui lòng đợi trong giây lát (khoảng 30 giây).")
    
    # 1. Load dữ liệu truyền thống (Traditional Data)
    app_train_path = os.path.join(input_dir, 'application_train.csv')
    if not os.path.exists(app_train_path):
        print(f"Lỗi: Không tìm thấy file {app_train_path}")
        return
    print("- Đang nạp application_train.csv...")
    app_train = pd.read_csv(app_train_path)
    
    # 2. Load và xử lý dữ liệu thay thế (Bureau Data)
    bureau_path = os.path.join(input_dir, 'bureau.csv')
    if not os.path.exists(bureau_path):
        print(f"Lỗi: Không tìm thấy file {bureau_path}")
        return
    print("- Đang nạp và tổng hợp bureau.csv...")
    bureau = pd.read_csv(bureau_path, usecols=['SK_ID_CURR', 'DAYS_CREDIT', 'SK_ID_BUREAU'])
    bureau_agg = bureau.groupby('SK_ID_CURR', as_index=False).agg({
        'DAYS_CREDIT': 'mean',
        'SK_ID_BUREAU': 'count'
    })
    bureau_agg.columns = ['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    
    # 3. Load và xử lý dữ liệu hồ sơ vay cũ (Previous Application Data)
    prev_path = os.path.join(input_dir, 'previous_application.csv')
    if os.path.exists(prev_path):
        print("- Đang nạp và tổng hợp previous_application.csv...")
        prev = pd.read_csv(prev_path, usecols=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT', 'NAME_CONTRACT_STATUS'])
        prev['IS_REJECTED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
        prev_agg = prev.groupby('SK_ID_CURR', as_index=False).agg({
            'SK_ID_PREV': 'count',
            'AMT_CREDIT': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
    else:
        print("- Cảnh báo: Không tìm thấy previous_application.csv, bỏ qua đặc trưng này.")
        prev_agg = pd.DataFrame(columns=['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT'])
        
    # 4. Load và xử lý lịch sử đóng tiền của khoản vay cũ (Installments Payments Data)
    inst_path = os.path.join(input_dir, 'installments_payments.csv')
    if os.path.exists(inst_path):
        print("- Đang nạp và tổng hợp installments_payments.csv...")
        inst = pd.read_csv(inst_path, usecols=['SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'])
        # Tính số ngày đóng trễ hạn (nếu DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)
        inst['PAY_DELAY'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
        # Tính số tiền còn nợ/thiếu của đợt đóng tiền
        inst['UNDERPAY'] = (inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']).clip(lower=0)
        
        inst_agg = inst.groupby('SK_ID_CURR', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
    else:
        print("- Cảnh báo: Không tìm thấy installments_payments.csv, bỏ qua đặc trưng này.")
        inst_agg = pd.DataFrame(columns=['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN'])
        
    # 5. Gộp tất cả các bảng dữ liệu lại với nhau (Merge)
    print("- Đang gộp các bảng dữ liệu...")
    df_combined = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    if not prev_agg.empty:
        df_combined = df_combined.merge(prev_agg, on='SK_ID_CURR', how='left')
    if not inst_agg.empty:
        df_combined = df_combined.merge(inst_agg, on='SK_ID_CURR', how='left')
        
    num_rows, num_cols = df_combined.shape
    
    # Tính tỷ lệ mất cân bằng TARGET
    target_counts = df_combined['TARGET'].value_counts(normalize=True) * 100
    percent_0 = target_counts.get(0, 0)
    percent_1 = target_counts.get(1, 0)
    
    # Lưu file đã làm sạch
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    print(f"- Đang xuất dữ liệu ra file '{output_path}'...")
    df_combined.to_csv(output_path, index=False)
    
    execution_duration = time.time() - start_time
    
    # In kết quả
    print("\n" + "="*60)
    print(f"Kích thước dữ liệu: In rõ số lượng hàng (~{num_rows:,}) và số cột mới (đã tăng lên từ 122 thành {num_cols} biến).")
    print(f"Tỷ lệ phần trăm: In rõ nhóm 0 chiếm khoảng {percent_0:.2f}% và nhóm 1 chiếm khoảng {percent_1:.2f}%.")
    print(f"File vật lý: Trong thư mục dự án của bạn sẽ tự động xuất hiện một thư mục mới tên là data_output chứa file cleaned_data.csv.")
    print(f"Thời gian thực thi: {execution_duration:.2f} giây")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
