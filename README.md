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
*   **`results/`**: Thư mục lưu trữ tự động các file log kết quả chạy dưới định dạng console của terminal (được gắn mã thời gian để không bị ghi đè).
*   **`plots/`**: Thư mục lưu trữ tự động biểu đồ độ quan trọng đặc trưng dạng ảnh (`feature_importance_xxx.png`).
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
    *   Đọc tệp `application_train.csv`.
    *   Tổng hợp dữ liệu số khoản vay ngoài từ `bureau.csv`.
    *   Tổng hợp số đơn vay quá khứ và số lần bị từ chối duyệt vay từ `previous_application.csv`.
    *   Tính trung bình số ngày trả trễ hạn (DPD) và số tiền đóng thiếu từ `installments_payments.csv`.
    *   Gộp toàn bộ thành file **`data_output/cleaned_data.csv`** (Tổng cộng **129 biến**).

### BƯỚC 2: Huấn luyện, tối ưu siêu tham số và so sánh mô hình
*   **Lệnh thực thi**:
    ```bash
    python step3_4_5_pipeline.py
    ```
*   **Logic hoạt động**:
    *   Đọc file `cleaned_data.csv`, xử lý giá trị dị biệt thâm niên làm việc (`DAYS_EMPLOYED`).
    *   Tự động tính toán các biến tỷ lệ tài chính mới (`DIR`, `AIR`, `ACR`, `DAR`) và biến tương tác tín dụng (`EXT_SOURCES_PROD`).
    *   Mã hóa categorical bằng `LabelEncoder` / `One-Hot Encoding` và điền giá trị thiếu tự động bằng `median`.
    *   Sửa lỗi tên cột chứa ký tự đặc biệt của LightGBM.
    *   Chia dữ liệu thành 2 tập: Train (80%) và Validation (20%).
    *   Huấn luyện và so sánh **Random Forest**, **LightGBM** (được tinh chỉnh siêu tham số), và **XGBoost** (được tinh chỉnh siêu tham số).
    *   Lấy mô hình có AUC tốt nhất, huấn luyện phiên bản cải tiến bằng cách thêm trọng số phạt `scale_pos_weight = 11.5` để cân bằng dữ liệu nợ xấu.
    *   Xuất biểu đồ độ quan trọng đặc trưng vào thư mục **`plots/`** và lưu toàn bộ báo cáo phân tích, công thức toán học vào thư mục **`results/`**.

### BƯỚC 3: Chạy dự đoán quyết định duyệt vay cho khách hàng mới (Chạy thực tế)
*   **Lệnh thực thi**:
    ```bash
    python predict_new_customers.py
    ```
*   **Logic hoạt động**:
    *   Hệ thống sẽ hỏi bạn đường dẫn file CSV chứa khách hàng mới (Ví dụ nhập: `new_customers/synthetic_new_customers_50.csv`).
    *   Nạp mô hình tốt nhất đã được huấn luyện.
    *   Chạy suy luận tính toán xác suất bùng nợ (%) cho từng khách hàng mới.
    *   Áp dụng ngưỡng cắt (Cut-off Threshold = 15%): Xác suất $\ge 15\% \rightarrow$ **TỪ CHỐI (Reject)**, Xác suất $< 15\% \rightarrow$ **PHÊ DUYỆT (Approve)**.
    *   Hiển thị bảng phê duyệt trực tiếp trên màn hình và xuất file báo cáo chi tiết vào thư mục **`results/`**.

---

## 3. Lý Thuyết & Công Thức Đóng Góp Khoa Học (Viết Luận Văn)

Để viết báo cáo NCKH, hệ thống đã chuẩn bị sẵn các công thức định dạng chuẩn toán học dưới đây ở cuối mỗi file kết quả:

### 3.1. Chỉ số AUC-ROC
Đo lường khả năng phân biệt khách hàng tốt/xấu của mô hình:
$$AUC = \int_{0}^{1} TPR(FPR^{-1}(x)) dx$$
Ý nghĩa: Xác suất mô hình xếp hạng điểm rủi ro khách hàng nợ xấu ngẫu nhiên cao hơn khách hàng tốt ngẫu nhiên.

### 3.2. Hệ số Gini Coefficient
Đo lường độ bất bình đẳng tập trung rủi ro:
$$Gini = 2 \times AUC - 1$$
Gini càng gần 1, mô hình phân tách rủi ro càng mạnh.

### 3.3. Tổn thất tài chính kỳ vọng (Expected Financial Loss - EFL)
Minh chứng hiệu quả tài chính thực tiễn khi đưa mô hình cải tiến vào áp dụng:
$$EFL = (FN \times D) + (FP \times C)$$
Trong đó:
*   `FN` (Lọt lưới nợ xấu): Thiệt hại trung bình là $D$ (ví dụ: 100 triệu VND).
*   `FP` (Từ chối nhầm khách tốt): Chi phí cơ hội mất lãi vay là $C$ (ví dụ: 10 triệu VND).
*   Mô hình cải tiến giúp tối thiểu hóa chỉ số EFL này để tiết kiệm chi phí cho ngân hàng.

### 3.4. Công thức biến số tài chính mới tự thiết kế
*   Tỷ lệ Nợ / Thu nhập: `DIR = AMT_CREDIT / AMT_INCOME_TOTAL`
*   Tỷ lệ Phí trả nợ định kỳ / Thu nhập: `AIR = AMT_ANNUITY / AMT_INCOME_TOTAL`
*   Tỷ lệ Phí trả nợ / Tổng nợ: `ACR = AMT_ANNUITY / AMT_CREDIT`
*   Tỷ lệ Số ngày làm việc / Độ tuổi: `DAR = DAYS_EMPLOYED / DAYS_BIRTH`
*   Biến tương tác tín dụng: `EXT_SOURCES_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3`
