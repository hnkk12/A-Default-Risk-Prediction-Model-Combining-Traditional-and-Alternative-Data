# Dự Án Nghiên Cứu Hệ Thống Chấm Điểm Tín Dụng & Dự Báo Rủi Ro Bùng Nợ (Credit Scoring Model)

Dự án này là một mô hình Học máy (Machine Learning) toàn diện phục vụ cho Nghiên cứu khoa học (NCKH) nhằm đánh giá rủi ro bùng nợ (Default Risk) của khách hàng vay cá nhân. Hệ thống kết hợp giữa **Dữ liệu hồ sơ truyền thống** (thông tin đăng ký) và **Dữ liệu hành vi thay thế** (lịch sử tín dụng Credit Bureau, lịch sử thanh toán trễ hạn và các đơn vay cũ).

---

## 1. Cấu Trúc Thư Mục Dự Án (Sau Khi Tối Ưu)

Để giữ cho dự án gọn gàng và tập trung vào số liệu thực nghiệm, thư mục đã được làm sạch và tổ chức như sau:

*   **`Input/`**: Chứa 5 tệp tin dữ liệu thô gốc bắt buộc của hệ thống:
    *   `application_train.csv`: Hồ sơ khách hàng quá khứ (dữ liệu học tập gốc).
    *   `application_test.csv`: Danh sách hồ sơ khách hàng mới (mặc định để dự báo).
    *   `bureau.csv`: Lịch sử tín dụng ngoài hệ thống (từ các tổ chức tài chính khác).
    *   `previous_application.csv`: Lịch sử các đơn vay cũ tại chính ngân hàng đó.
    *   `installments_payments.csv`: Lịch sử đóng tiền trả nợ định kỳ của khách hàng.
*   **`data_output/`**: Lưu trữ file `cleaned_data.csv` đã qua tích hợp và làm sạch.
*   **`new_customers/`**: Thư mục chứa các tệp tin CSV khách hàng mới để chạy dự báo:
    *   `template.csv`: Mẫu tối giản (5 cột cốt lõi).
    *   `synthetic_new_customers_50.csv`: Tệp kiểm thử 121 cột đầy đủ thông tin chuẩn hóa.
*   **`results/`**: Thư mục lưu trữ tự động các file log kết quả chạy dưới định dạng console của terminal (được gắn mã thời gian để không bị ghi đè, bao gồm số liệu thống kê và công thức toán học).
*   **`plots/`**: Thư mục lưu trữ tự động các biểu đồ dạng ảnh:
    *   `feature_importance_xxx.png`: Biểu đồ độ quan trọng đặc trưng toàn cục.
    *   `shap_summary_xxx.png`: Biểu đồ giải thích mô hình cục bộ và hướng tác động (SHAP Summary Plot).
*   **`data_update_guide.md`**: File hướng dẫn định dạng cột dữ liệu của Ngân hàng.
*   **Các file Python cốt lõi chạy pipeline**: `step1_2_data_processing.py`, `step3_4_5_pipeline.py`, `predict_new_customers.py`.

---

## 2. Quy Trình Vận Hành 3 Bước Lấy Số Liệu & Dự Báo

Bạn hãy mở terminal của mình tại thư mục dự án và chạy tuần tự các lệnh sau:

### BƯỚC 1: Sơ chế và tích hợp dữ liệu đa nguồn
*   **Lệnh thực thi**:
    ```bash
    python step1_2_data_processing.py
    ```
*   **Logic hoạt động**:
    *   Đọc tệp `application_train.csv` và các file phụ.
    *   Tổng hợp dữ liệu số khoản vay ngoài từ `bureau.csv`.
    *   Tổng hợp dữ liệu số đơn vay cũ và số lần bị từ chối từ `previous_application.csv`.
    *   Tính trung bình số ngày trả trễ hạn (DPD) và số tiền đóng thiếu từ `installments_payments.csv`.
    *   Gộp toàn bộ thành file **`data_output/cleaned_data.csv`** (Tổng cộng **129 biến**).

### BƯỚC 2: Huấn luyện nâng cao chéo, kiểm định thống kê và giải thích AI (Q1 Standard)
*   **Lệnh thực thi**:
    ```bash
    python step3_4_5_pipeline.py
    ```
*   **Logic hoạt động**:
    *   Đọc file `cleaned_data.csv`, xử lý dị biệt thâm niên công tác (`DAYS_EMPLOYED`).
    *   Tự động thiết kế các đặc trưng tỷ lệ tài chính (`DIR`, `AIR`, `ACR`, `DAR`) và biến tương tác tín dụng (`EXT_SOURCES_PROD`).
    *   Đo lường độ ổn định thông qua kiểm định chéo **3-Fold Stratified Cross-Validation**.
    *   Huấn luyện và so sánh chéo 4 thuật toán lớn: **Random Forest**, **LightGBM**, **XGBoost**, và **CatBoost** dựa trên 4 chỉ số: **AUC-ROC**, **Gini**, **Kolmogorov-Smirnov (KS)**, và **Brier Score** (đo lường độ hiệu chuẩn xác suất).
    *   Thực hiện kiểm định ý nghĩa thống kê (**Paired t-test**) giữa mô hình tốt nhất với các mô hình còn lại để tính toán trị số **p-value** chứng minh sự khác biệt.
    *   Áp dụng trọng số phạt `scale_pos_weight = 11.5` trên mô hình tốt nhất để cân bằng nhãn và đánh giá chi phí tổn thất kỳ vọng (EFL).
    *   Trích xuất độ quan trọng đặc trưng (`feature_importance_xxx.png`).
    *   Tính toán **SHAP values** bằng `shap.TreeExplainer` trên 500 mẫu đại diện và vẽ biểu đồ **SHAP Summary Plot** (`shap_summary_xxx.png`) để giải thích mô hình AI.
    *   Lưu toàn bộ báo cáo phân tích lý thuyết toán học vào thư mục **`results/`**.

### BƯỚC 3: Chạy dự đoán quyết định duyệt vay cho khách hàng mới (Chạy thực tế)
*   **Lệnh thực thi**:
    ```bash
    python predict_new_customers.py
    ```
*   **Logic hoạt động**:
    *   Hệ thống sẽ hỏi bạn đường dẫn file CSV chứa khách hàng mới (Nhập: `new_customers/synthetic_new_customers_50.csv` hoặc file của bạn).
    *   Nạp mô hình tốt nhất đã được huấn luyện.
    *   Chạy suy luận tính toán xác suất bùng nợ (%) và đưa ra quyết định phê duyệt cấp tín dụng dựa trên ngưỡng cắt rủi ro tối đa **15%**.

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
Độ lệch tối đa giữa hàm phân phối tích lũy (CDF) của nhóm Tốt và nhóm Xấu:
$$KS = \sup_{s} | F_{\text{good}}(s) - F_{\text{bad}}(s) |$$

### 3.4. Độ hiệu chuẩn xác suất Brier Score
$$Brier = \frac{1}{N} \sum_{i=1}^{N} (y_i - p_i)^2$$

### 3.5. Tổn thất tài chính kỳ vọng (Expected Financial Loss - EFL)
$$EFL = (FN \times D) + (FP \times C)$$
*(Với D là tổn thất trốn nợ, C là chi phí cơ hội từ chối nhầm)*

### 3.6. Định nghĩa toán học của biến Shapley (SHAP Values)
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$
*(Trong đó F là tập hợp tất cả các đặc trưng, S là tập hợp con không chứa đặc trưng i)*
