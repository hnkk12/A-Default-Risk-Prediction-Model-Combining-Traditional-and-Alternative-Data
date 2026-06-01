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
    
    # 1. Load hồ sơ chính
    app_train = pd.read_csv(os.path.join(input_dir, 'ho_so_khach_hang_train.csv'))
    
    # 2. Bureau (Tối giản)
    bureau_path = os.path.join(input_dir, 'lich_su_tin_dung_bureau.csv')
    if os.path.exists(bureau_path) and not pd.read_csv(bureau_path).empty:
        bureau = pd.read_csv(bureau_path)
        bureau_agg = bureau.groupby('ma_khach_hang', as_index=False).agg({
            'so_ngay_truoc_khi_vay_ngoai': 'mean',
            'ma_khoan_vay_ngoai': 'count'
        })
        bureau_agg.columns = ['ma_khach_hang', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    else:
        bureau_agg = pd.DataFrame(columns=['ma_khach_hang', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT'])
    
    # 3. Prev (Tối giản)
    prev_path = os.path.join(input_dir, 'don_vay_cu_noi_bo.csv')
    if os.path.exists(prev_path) and not pd.read_csv(prev_path).empty:
        prev = pd.read_csv(prev_path)
        prev['IS_REJECTED'] = (prev['ket_qua_duyet'] == 'Tu choi').astype(int)
        prev_agg = prev.groupby('ma_khach_hang', as_index=False).agg({
            'ma_don_vay_cu': 'count',
            'so_tien_duoc_duyet_vnd': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['ma_khach_hang', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
    else:
        prev_agg = pd.DataFrame(columns=['ma_khach_hang', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT'])
        
    # 4. Inst
    inst_path = os.path.join(input_dir, 'lich_su_tra_no_dinh_ky.csv')
    if os.path.exists(inst_path):
        inst = pd.read_csv(inst_path)
        inst['PAY_DELAY'] = (inst['ngay_thuc_te_tra'] - inst['ngay_phai_tra']).clip(lower=0)
        inst['UNDERPAY'] = (inst['so_tien_phai_tra_vnd'] - inst['so_tien_thuc_te_tra_vnd']).clip(lower=0)
        inst_agg = inst.groupby('ma_khach_hang', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['ma_khach_hang', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
    else:
        inst_agg = pd.DataFrame(columns=['ma_khach_hang', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN'])
        
    # 5. Merge
    df_combined = app_train.merge(bureau_agg, on='ma_khach_hang', how='left')
    df_combined = df_combined.merge(prev_agg, on='ma_khach_hang', how='left')
    df_combined = df_combined.merge(inst_agg, on='ma_khach_hang', how='left')
    
    output_path = os.path.join(output_dir, 'cleaned_data2.csv')
    df_combined.to_csv(output_path, index=False)
    
    print(f"Xuất file cleaned_data2.csv hoàn tất. Kích thước: {df_combined.shape}")

if __name__ == "__main__":
    main()
