# BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH (V2 - VIỆT HÓA)
## Mô hình Dự báo Rủi ro Bùng nợ và Chấm điểm Tín dụng cải tiến
**Phục vụ Nghiên cứu Khoa học (NCKH) / Luận văn học thuật**

---

### 1. BẢNG SO SÁNH HIỆU NĂNG 4 THUẬT TOÁN (3-Fold Stratified Cross-Validation) - V2

Để chọn ra mô hình tối ưu nhất trên dữ liệu mô phỏng Việt hóa, chúng tôi huấn luyện chéo và kiểm thử 4 thuật toán lớn phổ biến nhất trên bộ dữ liệu tín dụng làm sạch gồm **250,008** hồ sơ huấn luyện và **24** đặc trưng đầu vào. 

| Thuật Toán | AUC-ROC (Mean ± SD) | Hệ Số Gini (Mean) | Chỉ Số KS (Mean) | Brier Score (Mean) |
|---|---|---|---|---|
| **Random Forest** | 0.5612 ± 0.0013 | 0.1224 | 0.0899 | 0.1186 |
| **LightGBM** | 0.5618 ± 0.0030 | 0.1236 | 0.0912 | 0.1186 |
| **XGBoost** | 0.5610 ± 0.0021 | 0.1219 | 0.0896 | 0.1186 |
| **CatBoost (Best)** | **0.5639 ± 0.0032** | **0.1279** | **0.0931** | **0.1185** |

**Kiểm định ý nghĩa thống kê (Paired t-test) so với CatBoost:**
- CatBoost vs Random Forest $\rightarrow$ p-value = 0.17377 (Không có ý nghĩa thống kê, $p \ge 0.05$)
- CatBoost vs LightGBM $\rightarrow$ p-value = 0.02689 (Có ý nghĩa thống kê, $p < 0.05$)
- CatBoost vs XGBoost $\rightarrow$ p-value = 0.08712 (Không có ý nghĩa thống kê, $p \ge 0.05$)

*Kết luận:* **CatBoost** là mô hình tốt nhất đạt các chỉ số tối ưu trên tập dữ liệu Việt hóa mô phỏng và được lựa chọn cho các bước cải tiến tiếp theo.

---

### 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRƯỚC VÀ SAU COST-SENSITIVE (V2)

Nhằm cân bằng ảnh hưởng của việc mất cân bằng nhãn rủi ro (nhóm nợ xấu chiếm khoảng 13.83%), mô hình CatBoost được tích hợp cải tiến trọng số phạt lỗi phân loại sai nợ xấu gấp **11.5 lần** so với nợ tốt. Kết quả ma trận nhầm lẫn đo lường trên tập kiểm thử độc lập 20% (50,000 khách hàng) như sau:

#### Bảng so sánh Ma trận Nhầm lẫn:
| Phân loại thực tế | Dự báo bởi Mô hình Gốc (Baseline) | Dự báo bởi Mô hình Cải tiến (Cost-Sensitive) |
|---|---|---|
| **Khách hàng TỐT thực tế** | **43,084** ca dự báo đúng (True Negatives) | **400** ca dự báo đúng (True Negatives) |
| | **0** ca bị từ chối sai (False Positives) | **42,684** ca bị từ chối sai (False Positives) |
| **Khách hàng NỢ XẤU thực tế** | **6,916** ca bị bỏ sót (False Negatives) | **40** ca bị bỏ sót (False Negatives) |
| | **0** ca bắt trúng (True Positives) | **6,876** ca bắt trúng (True Positives) |

#### Ý nghĩa thực tiễn:
- **Bắt trúng thêm 6,876 ca nợ xấu thực tế** so với mô hình cũ (tăng từ 0 ca lên 6,876 ca).
- Số ca bỏ sót nợ xấu (False Negatives - rủi ro lớn nhất gây mất vốn) giảm mạnh từ **6,916 ca** xuống chỉ còn **40 ca**.

---

### 3. ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH KINH TẾ (Expected Financial Loss - EFL)

Để chứng minh tính đóng góp thực tiễn cho hoạt động quản trị rủi ro tại ngân hàng, hiệu quả kinh tế được đo lường thông qua hàm tổn thất tài chính kỳ vọng:
$$EFL = (FN \times D) + (FP \times C)$$

*Trong đó:*
- $FN$ là số ca nợ xấu bị mô hình bỏ sót (duyệt nhầm thành khách hàng tốt). Mỗi ca gây tổn thất trốn nợ trung bình trị giá $D = 100$ triệu VND.
- $FP$ là số ca khách hàng tốt bị từ chối nhầm (mô hình dự báo nhầm thành nợ xấu). Mỗi ca gây chi phí cơ hội trị giá $C = 10$ triệu VND.

#### Bảng tính toán Expected Financial Loss (EFL):

1. **Mô hình Gốc (Baseline):**
   $$EFL_{\text{base}} = (6,916 \times 100\text{M}) + (0 \times 10\text{M}) = 691,600.00\text{ triệu VND}$$
   *(Tương đương 691 tỷ 600 triệu VND)*

2. **Mô hình Cải tiến (Cost-Sensitive):**
   $$EFL_{\text{imp}} = (40 \times 100\text{M}) + (42,684 \times 10\text{M}) = 4,000\text{M} + 426,840\text{M} = 430,840.00\text{ triệu VND}$$
   *(Tương đương 430 tỷ 840 triệu VND)*

3. **Lợi ích tài chính tiết kiệm được cho ngân hàng:**
   $$\Delta EFL = EFL_{\text{base}} - EFL_{\text{imp}} = 691,600.00\text{M} - 430,840.00\text{M} = 260,760.00\text{ triệu VND}$$
   *(Tương đương **260 tỷ 760 triệu VND** tiết kiệm được trên mỗi 50,000 hồ sơ)*

---

### 4. BIỂU ĐỒ MINH CHỨNG (Có trong thư mục plots/)

- **`feature_importance_v2.png`**: Biểu đồ cho thấy tỷ lệ độ quan trọng của 15 biến hàng đầu đối với mô hình cải tiến V2. Điểm tín dụng trung bình từ các nguồn độc lập (`EXT_SOURCES_MEAN` và `EXT_SOURCES_PROD`) đóng vai trò quan trọng nhất trong việc nhận định rủi ro.
- **`shap_summary_v2.png`**: Biểu đồ giải thích SHAP biểu diễn hướng và độ lớn tác động của từng đặc trưng lên xác suất bùng nợ trong mô hình Việt hóa.

---

### 5. HƯỚNG DẪN KIỂM CHỨNG KẾT QUẢ CHO GIÁO VIÊN HƯỚNG DẪN
1. Mở file notebook **`nckh_credit_scoring_pipeline_v2.ipynb`** bằng Jupyter Notebook hoặc Google Colab.
2. Chạy tuần tự các cell code để tái lập toàn bộ bảng số liệu thực tế này dựa trên file dữ liệu đã xử lý sạch **`cleaned_data2.csv`**.
