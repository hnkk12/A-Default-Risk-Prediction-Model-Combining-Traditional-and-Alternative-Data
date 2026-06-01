import os
import pandas as pd
import numpy as np

def main():
    print("--- KHỞI TẠO DỮ LIỆU TÍN DỤNG VIỆT NAM (PHIÊN BẢN THỰC TẾ NCKH V6) ---")
    output_dir = 'Input2'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    np.random.seed(42)
    
    num_train, num_test = 15000, 3000
    total = num_train + num_test
    customer_ids = np.arange(5000001, 5000001 + total)
    targets = np.random.choice([0, 1], size=total, p=[0.92, 0.08])
    
    # 1. Điểm tín dụng: Làm cho 2 nhóm cực kỳ sát nhau để AUC không bị vọt lên
    # Độ lệch chuẩn (s) rất lớn (0.4) tạo ra sự hỗn loạn cần thiết
    def gen_noisy_scores(targets, m0, m1, s):
        raw = np.where(targets == 0, np.random.normal(m0, s, total), np.random.normal(m1, s, total))
        return (300 + np.clip(raw, 0, 1) * 550).astype(int)

    diem_1 = gen_noisy_scores(targets, 0.55, 0.48, 0.45)
    diem_2 = gen_noisy_scores(targets, 0.58, 0.50, 0.40)
    diem_3 = gen_noisy_scores(targets, 0.52, 0.45, 0.50)
    
    # 2. Biến tài chính: Overlap mạnh
    income = np.random.lognormal(mean=16.6, sigma=0.7, size=total)
    dir_ratio = np.where(targets == 0, np.random.normal(8, 6, total), np.random.normal(12, 10, total))
    dir_ratio = np.clip(dir_ratio, 1.0, 45.0)
    loan_amt = income * dir_ratio
    
    df = pd.DataFrame({
        'ma_khach_hang': customer_ids,
        'ho_va_ten': [f"Khach Hang {i}" for i in range(total)],
        'gioi_tinh': np.random.choice(["Nam", "Nu"], total),
        'loai_hop_dong': np.random.choice(["Vay tieu dung", "Vay mua nha", "Vay mua xe"], total),
        'co_o_to': np.random.choice(["Y", "N"], total),
        'co_nha_dat': np.random.choice(["Y", "N"], total),
        'so_con': np.random.randint(0, 4, total),
        'thu_nhap_nam_vnd': (income * 12).round(-4),
        'so_tien_vay_vnd': loan_amt.round(-5),
        'khoan_tra_dinh_ky_vnd': (loan_amt / 24 * 1.15).round(-3),
        'gia_tri_tai_san_mua_vnd': (loan_amt * 1.3).round(-5),
        'tuoi_khach_hang': np.random.randint(22, 68, total),
        'kinh_nghiem_lam_viec': np.random.randint(0, 30, total),
        'diem_tin_dung_1': diem_1,
        'diem_tin_dung_2': diem_2,
        'diem_tin_dung_3': diem_3
    })
    
    # 3. Lịch sử trả nợ (PAY_DELAY): Thủ phạm chính gây AUC=1.0
    # Ta sẽ làm cho biến này "nhiễu" hơn nữa. Nhiều khách xấu đóng rất chuẩn, nhiều khách tốt trễ triền miên.
    print("- Đang sinh lịch sử trả nợ có độ nhiễu cao (Realistic Overlap)...")
    payment_records = []
    for i in range(total):
        is_bad = targets[i]
        # Xáo trộn nhãn: 35% khách xấu sẽ có hành vi trả nợ giống hệt khách tốt và ngược lại
        if np.random.rand() < 0.35:
            effective_bad = 1 - is_bad
        else:
            effective_bad = is_bad
            
        p_lambda = np.random.uniform(1.0, 6.0) if effective_bad == 0 else np.random.uniform(5.0, 25.0)
        
        for m in range(1, 8):
            delay = np.random.poisson(p_lambda)
            # Thêm nhiễu Gaussian cực mạnh
            delay_final = max(0, delay + np.random.normal(0, 5.0)) 
            
            payment_records.append({
                'ma_khach_hang': customer_ids[i],
                'ma_don_vay_cu': 8000000 + i,
                'ky_han_thanh_toan': m,
                'ngay_phai_tra': -250 + m*30,
                'ngay_thuc_te_tra': -250 + m*30 + int(delay_final),
                'so_tien_phai_tra_vnd': 5000000,
                'so_tien_thuc_te_tra_vnd': 5000000 if (effective_bad == 0 or np.random.rand() > 0.5) else 2000000
            })
            
    df_payments = pd.DataFrame(payment_records)
    
    # Xuất dữ liệu
    df_train = df.iloc[:num_train].copy()
    df_train['TARGET'] = targets[:num_train]
    df_test = df.iloc[num_train:].copy()
    
    df_train.to_csv(os.path.join(output_dir, 'ho_so_khach_hang_train.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'ho_so_khach_hang_test.csv'), index=False)
    df_payments.to_csv(os.path.join(output_dir, 'lich_su_tra_no_dinh_ky.csv'), index=False)
    
    # Các file phụ khác
    pd.DataFrame(columns=['ma_khach_hang', 'ma_khoan_vay_ngoai', 'so_ngay_truoc_khi_vay_ngoai', 'trang_thai_khoan_vay']).to_csv(os.path.join(output_dir, 'lich_su_tin_dung_bureau.csv'), index=False)
    pd.DataFrame(columns=['ma_khach_hang', 'ma_don_vay_cu', 'so_tien_duoc_duyet_vnd', 'ket_qua_duyet']).to_csv(os.path.join(output_dir, 'don_vay_cu_noi_bo.csv'), index=False)
    
    print(f"\n[XONG] Bộ dữ liệu V6. AUC kỳ vọng: 0.72 - 0.78. Sẵn sàng cho NCKH.")

if __name__ == "__main__": main()
