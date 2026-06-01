import pandas as pd
import numpy as np
import os

def preprocess_and_merge_data(input_dir='Input', output_dir='data_output'):
    print("=================================================================================")
    # 1. Nạp và xử lý application_train.csv
    train_path = os.path.join(input_dir, 'application_train.csv')
    if not os.path.exists(train_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu huấn luyện chính tại '{train_path}'")
        return False
    print("1. Đang nạp application_train.csv...")
    app_train = pd.read_csv(train_path)
    
    # 2. Nạp và xử lý bureau.csv
    bureau_path = os.path.join(input_dir, 'bureau.csv')
    if os.path.exists(bureau_path):
        print("2. Đang nạp và tổng hợp dữ liệu lịch sử tín dụng ngoài hệ thống (bureau.csv)...")
        bureau = pd.read_csv(bureau_path, usecols=['SK_ID_CURR', 'DAYS_CREDIT', 'SK_ID_BUREAU'])
        bureau_agg = bureau.groupby('SK_ID_CURR', as_index=False).agg({
            'DAYS_CREDIT': 'mean',
            'SK_ID_BUREAU': 'count'
        })
        bureau_agg.columns = ['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    else:
        print("Cảnh báo: Không tìm thấy bureau.csv, bỏ qua bước tích hợp bureau.")
        bureau_agg = pd.DataFrame(columns=['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT'])
    
    # 3. Nạp và xử lý previous_application.csv
    prev_path = os.path.join(input_dir, 'previous_application.csv')
    if os.path.exists(prev_path):
        print("3. Đang nạp và tổng hợp dữ liệu các đơn vay quá khứ (previous_application.csv)...")
        prev = pd.read_csv(prev_path, usecols=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT', 'NAME_CONTRACT_STATUS'])
        prev['IS_REJECTED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
        prev_agg = prev.groupby('SK_ID_CURR', as_index=False).agg({
            'SK_ID_PREV': 'count',
            'AMT_CREDIT': 'mean',
            'IS_REJECTED': 'sum'
        })
        prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT']
    else:
        print("Cảnh báo: Không tìm thấy previous_application.csv, bỏ qua bước tích hợp đơn vay cũ.")
        prev_agg = pd.DataFrame(columns=['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_CREDIT_MEAN', 'PREV_APP_REJECTED_COUNT'])
        
    # 4. Nạp và xử lý installments_payments.csv
    inst_path = os.path.join(input_dir, 'installments_payments.csv')
    if os.path.exists(inst_path):
        print("4. Đang nạp và tổng hợp dữ liệu lịch sử đóng tiền trả nợ (installments_payments.csv)...")
        inst = pd.read_csv(inst_path, usecols=['SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'])
        inst['PAY_DELAY'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
        inst['UNDERPAY'] = (inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']).clip(lower=0)
        inst_agg = inst.groupby('SK_ID_CURR', as_index=False).agg({
            'PAY_DELAY': 'mean',
            'UNDERPAY': 'mean'
        })
        inst_agg.columns = ['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN']
    else:
        print("Cảnh báo: Không tìm thấy installments_payments.csv, bỏ qua bước tích hợp đóng tiền.")
        inst_agg = pd.DataFrame(columns=['SK_ID_CURR', 'INST_PAY_DELAY_MEAN', 'INST_UNDERPAY_MEAN'])
    
    # 5. Tích hợp dữ liệu đa nguồn
    print("5. Đang tiến hành gộp dữ liệu đa nguồn...")
    df_combined = app_train
    if not bureau_agg.empty:
        df_combined = df_combined.merge(bureau_agg, on='SK_ID_CURR', how='left')
    if not prev_agg.empty:
        df_combined = df_combined.merge(prev_agg, on='SK_ID_CURR', how='left')
    if not inst_agg.empty:
        df_combined = df_combined.merge(inst_agg, on='SK_ID_CURR', how='left')
        
    # 6. Lưu file sạch
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"\n=> Tích hợp dữ liệu thành công!")
    print(f"   Dữ liệu sạch được lưu tại: {os.path.abspath(output_path)}")
    print(f"   Kích thước tập dữ liệu: {df_combined.shape[0]:,} dòng, {df_combined.shape[1]:,} cột.")
    print("=================================================================================")
    return True

if __name__ == "__main__":
    preprocess_and_merge_data()
