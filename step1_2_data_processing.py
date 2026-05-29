import pandas as pd
import os
import time

# Bước 1.1: Tạo file code Python
# Tên file: step1_2_data_processing.py

def main():
    start_time = time.time()
    
    # Bước 1.2: Chuẩn bị dữ liệu đầu vào
    input_dir = 'input'
    output_dir = 'data_output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Đang xử lý dữ liệu... Vui lòng đợi trong giây lát.")
    
    # Load dữ liệu truyền thống (Traditional Data)
    # application_train.csv chứa thông tin hồ sơ hiện tại
    app_train_path = os.path.join(input_dir, 'application_train.csv')
    if not os.path.exists(app_train_path):
        print(f"Lỗi: Không tìm thấy file {app_train_path}")
        return
    app_train = pd.read_csv(app_train_path)
    
    # Load dữ liệu thay thế (Alternative Data)
    # bureau.csv chứa lịch sử tín dụng tại các tổ chức khác
    bureau_path = os.path.join(input_dir, 'bureau.csv')
    if not os.path.exists(bureau_path):
        print(f"Lỗi: Không tìm thấy file {bureau_path}")
        return
    bureau = pd.read_csv(bureau_path)
    
    # Xử lý dữ liệu bureau: Tính toán các đặc trưng tổng hợp (Aggregation)
    # Chúng ta tính 2 đặc trưng quan trọng để bổ sung vào hồ sơ khách hàng:
    # 1. BURO_COUNT: Số lượng khoản vay cũ của khách hàng tại các tổ chức tín dụng khác.
    # 2. BURO_DAYS_CREDIT_MEAN: Trung bình số ngày từ lúc đăng ký khoản vay cũ đến khi nộp hồ sơ hiện tại.
    bureau_agg = bureau.groupby('SK_ID_CURR', as_index=False).agg({
        'DAYS_CREDIT': 'mean',
        'SK_ID_BUREAU': 'count'
    })
    
    # Đổi tên cột để tránh trùng lặp và rõ nghĩa hơn
    bureau_agg.columns = ['SK_ID_CURR', 'BURO_DAYS_CREDIT_MEAN', 'BURO_COUNT']
    
    # Gộp dữ liệu truyền thống và dữ liệu thay thế (Merge)
    # Sử dụng left join theo SK_ID_CURR để giữ nguyên 307,511 hồ sơ khách hàng hiện tại
    df_combined = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Điền giá trị thiếu cho các khách hàng không có thông tin trong bureau (nếu cần)
    # Ở đây chúng ta giữ nguyên để khớp với yêu cầu về số lượng cột (122 + 2 = 124)
    
    # Thu thập thông tin kết quả
    num_rows, num_cols = df_combined.shape
    
    # Tính tỷ lệ phần trăm các nhóm trong TARGET
    target_counts = df_combined['TARGET'].value_counts(normalize=True) * 100
    percent_0 = target_counts.get(0, 0)
    percent_1 = target_counts.get(1, 0)
    
    # Lưu file kết quả vào thư mục data_output
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df_combined.to_csv(output_path, index=False)
    
    end_time = time.time()
    execution_duration = end_time - start_time
    
    # In kết quả ra màn hình theo định dạng yêu cầu
    print("\n" + "="*60)
    print(f"Kích thước dữ liệu: In rõ số lượng hàng (~{num_rows:,}) và số cột mới (đã tăng lên từ 122 thành {num_cols} biến).")
    print(f"Tỷ lệ phần trăm: In rõ nhóm 0 chiếm khoảng {percent_0:.2f}% và nhóm 1 chiếm khoảng {percent_1:.2f}%.")
    print(f"File vật lý: Trong thư mục dự án của bạn sẽ tự động xuất hiện một thư mục mới tên là data_output chứa file cleaned_data.csv.")
    print(f"Thời gian thực thi: {execution_duration:.2f} giây")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
