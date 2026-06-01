# Dự Án Nghiên Cứu Hệ Thống Chấm Điểm Tín Dụng & Dự Báo Rủi Ro Bùng Nợ (Credit Scoring Model)

Dự án này xây dựng một mô hình Học máy (Machine Learning) toàn diện phục vụ Nghiên cứu khoa học (NCKH) và luận văn học thuật, nhằm đánh giá rủi ro bùng nợ (Default Risk) của khách hàng cá nhân. Hệ thống kết hợp giữa **Dữ liệu hồ sơ truyền thống** (thông tin đăng ký cơ bản) và **Dữ liệu hành vi thay thế** (lịch sử tín dụng Credit Bureau, lịch sử thanh toán trễ hạn và các đơn vay cũ).

---

## 1. Cấu Trúc Thư Mục Dự Án (Sau Khi Tối Ưu)

Thư mục đã được làm sạch hoàn toàn các phiên bản cũ để giữ cho dự án gọn gàng và tập trung vào số liệu thực nghiệm khoa học:

*   **`Input/`**: Chứa 5 tệp tin dữ liệu thô gốc bắt buộc của hệ thống (được ẩn trên Git do dung lượng lớn >1.4 GB):
    *   `application_train.csv`: Hồ sơ khách hàng quá khứ (dữ liệu học tập gốc).
    *   `application_test.csv`: Danh sách hồ sơ khách hàng mới (mặc định để dự báo).
    *   `bureau.csv`: Lịch sử tín dụng ngoài hệ thống (từ các tổ chức tín dụng khác).
    *   `previous_application.csv`: Lịch sử các đơn vay cũ tại chính ngân hàng.
    *   `installments_payments.csv`: Lịch sử đóng tiền trả nợ định kỳ.
*   **`data_output/`**: Lưu trữ tệp tin `cleaned_data.csv` sau khi đã qua tích hợp đa nguồn và làm sạch.
*   **`new_customers/`**: Chứa tệp tin mẫu kiểm thử khách hàng mới:
    *   `synthetic_new_customers_50.csv`: Tệp kiểm thử gồm 50 khách hàng với đầy đủ thông tin chuẩn hóa để phản biện/người dùng chạy thử trực tiếp.
*   **`results/`**: Thư mục lưu trữ tự động các file log kết quả chạy dưới định dạng console của terminal (được gắn mã thời gian, bao gồm số liệu thống kê và công thức toán học).
*   **`plots/`**: Thư mục lưu trữ tự động các biểu đồ dạng ảnh:
    *   `feature_importance_xxx.png`: Biểu đồ độ quan trọng đặc trưng toàn cục.
    *   `shap_summary_xxx.png`: Biểu đồ giải thích mô hình cục bộ và hướng tác động (SHAP Summary Plot).
*   **`MinhChung_KetQua_ChayThat/`**: Thư mục lưu trữ tài liệu chứng minh thực tế tĩnh phục vụ nộp hội nghị (bao gồm báo cáo tóm tắt, biểu đồ mẫu, và file console log tĩnh).
*   **Các file Python cốt lõi chạy pipeline**: `step1_2_data_processing.py`, `step3_4_5_pipeline.py`, `predict_new_customers.py`.

---

## 2. Quy Trình Vận Hành 3 Bước Lấy Số Liệu & Dự Báo

Mở terminal tại thư mục dự án và chạy tuần tự các lệnh sau:

### BƯỚC 1: Sơ chế và tích hợp dữ liệu đa nguồn
*   **Lệnh thực thi**:
    ```bash
    python step1_2_data_processing.py
    ```
*   **Logic hoạt động**:
    *   Đọc tệp dữ liệu huấn luyện chính `application_train.csv` và gộp thông tin với các file phụ.
    *   Tổng hợp dữ liệu số khoản vay ngoài từ `bureau.csv` (`BURO_DAYS_CREDIT_MEAN`, `BURO_COUNT`).
    *   Tổng hợp lịch sử đơn vay cũ và số lần bị từ chối từ `previous_application.csv` (`PREV_APP_COUNT`, `PREV_APP_CREDIT_MEAN`, `PREV_APP_REJECTED_COUNT`).
    *   Tính trung bình số ngày trả trễ hạn (DPD) và số tiền đóng thiếu từ `installments_payments.csv` (`INST_PAY_DELAY_MEAN`, `INST_UNDERPAY_MEAN`).
    *   Gộp toàn bộ thành file **`data_output/cleaned_data.csv`**.

### BƯỚC 2: Huấn luyện nâng cao chéo, kiểm định thống kê và giải thích AI (Q1 Standard)
*   **Lệnh thực thi**:
    ```bash
    python step3_4_5_pipeline.py
    ```
*   **Logic hoạt động**:
    *   Đọc file `cleaned_data.csv`, xử lý các giá trị bất thường về thâm niên (`DAYS_EMPLOYED`).
    *   Thiết kế các đặc trưng tỷ lệ tài chính (`DIR`, `AIR`, `ACR`, `DAR`) và biến tương tác tín dụng (`EXT_SOURCES_PROD`, `EXT_SOURCES_MEAN`, `EXT_SOURCE_2_3_MULT`).
    *   Đo lường độ ổn định thông qua kiểm định chéo **3-Fold Stratified Cross-Validation**.
    *   Huấn luyện và so sánh chéo 4 thuật toán lớn: **Random Forest**, **LightGBM**, **XGBoost**, và **CatBoost** dựa trên 4 chỉ số: **AUC-ROC**, **Gini**, **Kolmogorov-Smirnov (KS)**, và **Brier Score**.
    *   Thực hiện kiểm định ý nghĩa thống kê (**Paired t-test**) giữa mô hình tốt nhất với các mô hình còn lại để tính toán trị số **p-value**.
    *   Áp dụng trọng số phạt `scale_pos_weight = 11.5` trên mô hình tốt nhất để cân bằng nhãn và đánh giá chi phí tổn thất kỳ vọng (EFL).
    *   Xuất biểu đồ độ quan trọng đặc trưng (`feature_importance_xxx.png`).
    *   Tính toán **SHAP values** bằng `shap.TreeExplainer` trên 500 mẫu đại diện và vẽ biểu đồ **SHAP Summary Plot** (`shap_summary_xxx.png`).
    *   Lưu toàn bộ báo cáo phân tích lý thuyết toán học vào thư mục **`results/`**.

### BƯỚC 3: Chạy dự đoán quyết định duyệt vay cho khách hàng mới
*   **Lệnh thực thi**:
    ```bash
    python predict_new_customers.py
    ```
*   **Logic hoạt động**:
    *   Hệ thống sẽ yêu cầu nhập đường dẫn file CSV chứa khách hàng mới cần duyệt vay (Nhập mẫu: `new_customers/synthetic_new_customers_50.csv` hoặc ấn Enter để dùng file mặc định).
    *   Huấn luyện mô hình tối ưu nhất trên toàn bộ dữ liệu lịch sử.
    *   Tự động tích hợp và tính toán các đặc trưng trên dữ liệu khách hàng mới.
    *   Chạy suy luận xác suất bùng nợ (%) và đưa ra quyết định phê duyệt cấp tín dụng dựa trên ngưỡng cắt rủi ro tối đa **15%**.

---

## 3. Lý Thuyết & Công Thức Đóng Góp Khoa Học (Viết Luận Văn Q1)

Để phục vụ viết báo cáo NCKH chuẩn quốc tế, hệ thống đã chuẩn bị sẵn các công thức định dạng chuẩn toán học LaTeX ở cuối mỗi file kết quả:

### 3.1. Chỉ số AUC-ROC
Đo lường khả năng phân biệt khách hàng tốt/xấu của mô hình:
$$AUC = \int_{0}^{1} TPR(FPR^{-1}(x)) dx$$
Xác suất lý thuyết: $AUC = P(f(x_{\text{bad}}) > f(x_{\text{good}}))$

### 3.2. Hệ số Gini Coefficient
$$Gini = 2 \times AUC - 1$$

### 3.3. Chỉ số Kolmogorov-Smirnov (KS)
Đo lường độ lệch tối đa giữa hàm phân phối tích lũy (CDF) của nhóm Tốt và nhóm Xấu:
$$KS = \sup_{s} | F_{\text{good}}(s) - F_{\text{bad}}(s) |$$

### 3.4. Độ hiệu chuẩn xác suất Brier Score
$$Brier = \frac{1}{N} \sum_{i=1}^{N} (y_i - p_i)^2$$

### 3.5. Tổn thất tài chính kỳ vọng (Expected Financial Loss - EFL)
$$EFL = (FN \times D) + (FP \times C)$$
*(Với D là tổn thất thực tế khi bỏ sót nợ xấu, C là chi phí cơ hội từ chối nhầm)*

### 3.6. Định nghĩa toán học của biến Shapley (SHAP Values)
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$
*(Trong đó F là tập hợp tất cả các đặc trưng, S là tập hợp con không chứa đặc trưng i)*
