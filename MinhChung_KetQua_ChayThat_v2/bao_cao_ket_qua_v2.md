# BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH (V2)
## Mô hình Dự báo Rủi ro Bùng nợ và Chấm điểm Tín dụng cải tiến - Dữ liệu Việt hóa
**Phục vụ Nghiên cứu Khoa học (NCKH) / Luận văn học thuật**

---

### 1. BẢNG SO SÁNH HIỆU NĂNG 4 THUẬT TOÁN (3-Fold Stratified Cross-Validation)

| Thuật Toán | AUC-ROC (Mean ± SD) | Hệ Số Gini (Mean) | Chỉ Số KS (Mean) | Brier Score (Mean) |
|---|---|---|---|---|
| Random Forest | 0.7342 ± 0.0159 | 0.4683 | 0.3519 | 0.0648 |
| LightGBM | 0.7282 ± 0.0142 | 0.4563 | 0.3450 | 0.0654 |
| XGBoost | 0.7328 ± 0.0125 | 0.4655 | 0.3496 | 0.0649 |
| **CatBoost** | 0.7486 ± 0.0207 | 0.4972 | 0.3766 | 0.0637 |

---

### 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRƯỚC VÀ SAU CẢI TIẾN

| Phân loại thực tế | Dự báo bởi Mô hình Gốc (Baseline) | Dự báo bởi Mô hình Cải tiến (Cost-Sensitive) |
|---|---|---|
| **Khách hàng TỐT thực tế** | **2,752** ca dự báo đúng (TN) | **2,414** ca dự báo đúng (TN) |
| | **7** ca bị từ chối sai (FP) | **345** ca bị từ chối sai (FP) |
| **Khách hàng NỢ XẤU thực tế** | **210** ca bị bỏ sót (FN) | **141** ca bị bỏ sót (FN) |
| | **31** ca bắt trúng (TP) | **100** ca bắt trúng (TP) |

#### Ý nghĩa thực tiễn:
- **Bắt trúng thêm 69 ca nợ xấu thực tế** so với mô hình cũ.
- Số ca bỏ sót nợ xấu (False Negatives) giảm từ **210 ca** xuống còn **141 ca**.

---

### 3. ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH KINH TẾ (Expected Financial Loss - EFL)
$$EFL = (FN \times 100M) + (FP \times 10M)$$

1. **Mô hình Gốc (Baseline):** 21,070.00 triệu VND
2. **Mô hình Cải tiến:** 17,550.00 triệu VND
3. **Lợi ích tài chính tiết kiệm được:** **3,520.00 triệu VND**

---

### 4. BIỂU ĐỒ MINH CHỨNG
- `plots/feature_importance_v2.png`
- `plots/shap_summary_v2.png`
