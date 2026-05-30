# BÁO CÁO KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH
## Mô hình Dự báo Rủi ro Bùng nợ và Chấm điểm Tín dụng cải tiến
**Phục vụ Nghiên cứu Khoa học (NCKH) / Luận văn học thuật**

---

### 1. BẢNG SO SÁNH HIỆU NĂNG 4 THUẬT TOÁN (3-Fold Stratified Cross-Validation)

Để chọn ra mô hình tối ưu nhất, chúng tôi huấn luyện chéo và kiểm thử 4 thuật toán lớn phổ biến nhất trên bộ dữ liệu tín dụng làm sạch gồm **246,008** hồ sơ huấn luyện và **233** đặc trưng đầu vào. 

| Thuật Toán | AUC-ROC (Mean ± SD) | Hệ Số Gini (Mean) | Chỉ Số KS (Mean) | Brier Score (Mean) |
|---|---|---|---|---|
| **Random Forest** | 0.7458 ± 0.0021 | 0.4917 | 0.3665 | 0.0689 |
| **LightGBM (Best)** | **0.7680 ± 0.0013** | **0.5360** | **0.4017** | **0.0673** |
| **XGBoost** | 0.7663 ± 0.0015 | 0.5326 | 0.3979 | 0.0674 |
| **CatBoost** | 0.7588 ± 0.0014 | 0.5175 | 0.3879 | 0.0680 |

**Kiểm định ý nghĩa thống kê (Paired t-test) so với LightGBM:**
- LightGBM vs Random Forest $\rightarrow$ p-value = 0.00171 (Có ý nghĩa thống kê cực kỳ rõ nét, $p < 0.01$)
- LightGBM vs XGBoost $\rightarrow$ p-value = 0.00652 (Có ý nghĩa thống kê rõ nét, $p < 0.01$)
- LightGBM vs CatBoost $\rightarrow$ p-value = 0.00015 (Có ý nghĩa thống kê cực kỳ rõ nét, $p < 0.001$)

*Kết luận:* **LightGBM** là mô hình tốt nhất với các chỉ số tối ưu tuyệt đối và có sự khác biệt vượt trội mang ý nghĩa thống kê thực nghiệm.

---

### 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TRƯỚC VÀ SAU COST-SENSITIVE

Nhằm cân bằng ảnh hưởng của việc mất cân bằng nhãn rủi ro (nhóm nợ xấu chỉ chiếm 8.07%), mô hình LightGBM được tích hợp cải tiến trọng số phạt lỗi phân loại sai nợ xấu gấp **11.5 lần** so với nợ tốt. Kết quả ma trận nhầm lẫn đo lường trên tập kiểm thử độc lập 20% (61,502 khách hàng) như sau:

#### Bảng so sánh Ma trận Nhầm lẫn:
| Phân loại thực tế | Dự báo bởi Mô hình Gốc (Baseline) | Dự báo bởi Mô hình Cải tiến (Cost-Sensitive) |
|---|---|---|
| **Khách hàng TỐT thực tế** | **56,496** ca dự báo đúng (True Negatives) | **39,688** ca dự báo đúng (True Negatives) |
| | **42** ca bị từ chối sai (False Positives) | **16,850** ca bị từ chối sai (False Positives) |
| **Khách hàng NỢ XẤU thực tế** | **4,891** ca bị bỏ sót (False Negatives) | **1,473** ca bị bỏ sót (False Negatives) |
| | **74** ca bắt trúng (True Positives) | **3,492** ca bắt trúng (True Positives) |

#### Ý nghĩa thực tiễn:
- **Bắt trúng thêm 3,418 ca nợ xấu thực tế** so với mô hình cũ (tăng từ 74 ca lên 3,492 ca, tức là tăng hiệu suất nhận diện nợ xấu lên gấp 47 lần).
- Số ca bỏ sót nợ xấu (False Negatives - rủi ro lớn nhất của ngân hàng) giảm mạnh từ **4,891 ca** xuống chỉ còn **1,473 ca**.

---

### 3. ĐÁNH GIÁ HIỆU QUẢ TÀI CHÍNH KINH TẾ (Expected Financial Loss - EFL)

Để chứng minh tính đóng góp thực tiễn cho hoạt động quản trị rủi ro tại ngân hàng, hiệu quả kinh tế được đo lường thông qua hàm tổn thất tài chính kỳ vọng:
$$EFL = (FN \times D) + (FP \times C)$$

*Trong đó:*
- $FN$ là số ca nợ xấu bị mô hình bỏ sót (duyệt nhầm thành khách hàng tốt). Mỗi ca gây tổn thất trốn nợ trung bình trị giá $D = 100$ triệu VND.
- $FP$ là số ca khách hàng tốt bị từ chối nhầm (mô hình dự báo nhầm thành nợ xấu). Mỗi ca gây chi phí cơ hội trị giá $C = 10$ triệu VND (lợi nhuận biên ròng của khoản vay bị mất).

#### Bảng tính toán Expected Financial Loss (EFL):

1. **Mô hình Gốc (Baseline):**
   $$EFL_{\text{base}} = (4,891 \times 100\text{M}) + (42 \times 10\text{M}) = 489,100\text{M} + 420\text{M} = 489,520.00\text{ triệu VND}$$
   *(Tương đương 489 tỷ 520 triệu VND)*

2. **Mô hình Cải tiến (Cost-Sensitive):**
   $$EFL_{\text{imp}} = (1,473 \times 100\text{M}) + (16,850 \times 10\text{M}) = 147,300\text{M} + 168,500\text{M} = 315,800.00\text{ triệu VND}$$
   *(Tương đương 315 tỷ 800 triệu VND)*

3. **Lợi ích tài chính tiết kiệm được cho ngân hàng:**
   $$\Delta EFL = EFL_{\text{base}} - EFL_{\text{imp}} = 489,520.00\text{M} - 315,800.00\text{M} = 173,720.00\text{ triệu VND}$$
   *(Tương đương **173 tỷ 720 triệu VND** tiết kiệm được trên mỗi 61,502 hồ sơ)*

---

### 4. BIỂU ĐỒ MINH CHỨNG (Có trong thư mục plots/)

- **`feature_importance.png`**: Biểu đồ cho thấy tỷ lệ độ quan trọng của 15 biến hàng đầu. Cột nợ phải trả định kỳ chia cho dư nợ (ACR), tuổi đời khách hàng (DAYS_BIRTH), và các nguồn điểm tín dụng bên thứ ba (EXT_SOURCE_*) có mức độ quan trọng cao nhất.
- **`shap_summary.png`**: Biểu đồ giải thích SHAP biểu diễn hướng và độ lớn tác động của từng đặc trưng. Ví dụ, điểm tín dụng bên thứ ba càng thấp (màu xanh dương) càng làm tăng xác suất bùng nợ (SHAP value dương lớn).

---

### 5. HƯỚNG DẪN KIỂM CHỨNG KẾT QUẢ CHO GIÁO VIÊN HƯỚNG DẪN
1. Mở file notebook **`nckh_credit_scoring_pipeline.ipynb`** bằng Jupyter Notebook, JupyterLab hoặc Google Colab.
2. Chạy tuần tự các cell code để tái lập toàn bộ bảng số liệu thực tế này dựa trên file dữ liệu đã xử lý sạch **`cleaned_data.csv`**.
