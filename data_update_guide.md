# Hướng Dẫn Định Dạng Dữ Liệu & Cập Nhật Hệ Thống (Dành Cho Ngân Hàng Thương Mại)

Tài liệu này hướng dẫn chi tiết cách chuẩn bị cấu trúc tệp tin dữ liệu đầu vào (Data Schema) và quy trình cập nhật hoặc huấn luyện lại mô hình khi ngân hàng phát sinh dữ liệu của chu kỳ kinh doanh mới.

---

## 1. ĐỊNH DẠNG CÁC TỆP TIN DỮ LIỆU ĐẦU VÀO (INPUT SCHEMA)

Để hệ thống hoạt động chính xác và không gặp lỗi cấu trúc (schema error), các tệp tin CSV đặt trong thư mục **`Input/`** phải tuân thủ nghiêm ngặt định dạng cột sau đây:

### 1.1. Tệp tin hồ sơ khách hàng (`application_train.csv` & `application_test.csv`)
*   **Định dạng**: File CSV, sử dụng dấu phẩy `,` làm dấu phân cách.
*   **Sự khác biệt**: 
    *   `application_train.csv` (dùng để dạy học/huấn luyện): Bắt buộc phải có cột **`TARGET`** (nhãn rủi ro: `0` là trả tốt, `1` là nợ xấu).
    *   `application_test.csv` (dùng để dự đoán khách hàng mới mặc định): **Không có cột `TARGET`**.
*   **Các trường thông tin cốt lõi bắt buộc**:

| Tên Cột | Kiểu Dữ Liệu | Ý Nghĩa Nghiệp Vụ | Ví Dụ |
| :--- | :--- | :--- | :--- |
| `SK_ID_CURR` | Số nguyên (`int`) | Mã định danh duy nhất của hồ sơ khách hàng | `100002` |
| `AMT_INCOME_TOTAL`| Số thực (`float`) | Tổng thu nhập hàng năm của khách hàng | `135000.0` |
| `AMT_CREDIT` | Số thực (`float`) | Số tiền khách hàng đăng ký vay | `450000.0` |
| `AMT_ANNUITY` | Số thực (`float`) | Số tiền khách hàng phải trả định kỳ (gốc + lãi) | `25000.0` |
| `DAYS_BIRTH` | Số nguyên (`int`) | Độ tuổi khách hàng (tính bằng số ngày âm tính ngược từ hiện tại) | `-12000` (~33 tuổi) |
| `DAYS_EMPLOYED` | Số nguyên (`int`) | Thâm niên làm việc tại đơn vị hiện tại (số ngày âm) | `-1500` (~4 năm) |
| `EXT_SOURCE_1, 2, 3`| Số thực (`float`) | Điểm tín dụng từ các đối tác hoặc CIC bên thứ ba (từ 0.0 đến 1.0) | `0.556`, `0.623` |

---

### 1.2. Tệp tin lịch sử tín dụng ngoài hệ thống (`bureau.csv`)
*   **Vai trò**: Cung cấp thông tin khách hàng đang nợ bao nhiêu khoản ở nơi khác.
*   **Các trường bắt buộc**:
    *   `SK_ID_CURR` (Số nguyên): Để liên kết với mã khách hàng chính.
    *   `DAYS_CREDIT` (Số nguyên): Số ngày kể từ khi đăng ký khoản vay ngoài.
    *   `SK_ID_BUREAU` (Số nguyên): Mã định danh duy nhất của khoản vay ngoài.

---

### 1.3. Tệp tin lịch sử đơn vay cũ trong hệ thống (`previous_application.csv`)
*   **Vai trò**: Cung cấp thông tin lịch sử giao dịch của khách hàng tại chính ngân hàng đó.
*   **Các trường bắt buộc**:
    *   `SK_ID_CURR` (Số nguyên): Liên kết với mã khách hàng chính.
    *   `SK_ID_PREV` (Số nguyên): Mã khoản vay cũ.
    *   `AMT_CREDIT` (Số thực): Số tiền vay được phê duyệt của các khoản vay cũ.
    *   `NAME_CONTRACT_STATUS` (Chữ): Trạng thái đơn vay cũ (ví dụ: `Approved`, `Refused`, `Canceled`).

---

### 1.4. Tệp tin lịch sử đóng tiền định kỳ (`installments_payments.csv`)
*   **Vai trò**: Cung cấp thói quen trả nợ (trả trễ hạn, trả thiếu tiền).
*   **Các trường bắt buộc**:
    *   `SK_ID_CURR` (Số nguyên): Liên kết với mã khách hàng chính.
    *   `DAYS_INSTALMENT` (Số nguyên): Ngày đến hạn thanh toán theo hợp đồng.
    *   `DAYS_ENTRY_PAYMENT` (Số nguyên): Ngày thực tế khách hàng đóng tiền.
    *   `AMT_INSTALMENT` (Số thực): Số tiền bắt buộc phải đóng theo đợt.
    *   `AMT_PAYMENT` (Số thực): Số tiền thực tế khách hàng đóng.

---

## 2. QUY TRÌNH 4 BƯỚC ĐỂ CẬP NHẬT MÔ HÌNH (MODEL RETRAINING)

Khi ngân hàng có dữ liệu của chu kỳ kinh doanh mới (ví dụ sau 6 hoặc 12 tháng) và muốn huấn luyện lại mô hình để bắt kịp xu hướng rủi ro mới:

### BƯỚC 1: Cập nhật file dữ liệu mới vào đúng vị trí
*   Trích xuất dữ liệu mới từ Database hệ thống Core Banking theo đúng cấu trúc cột ở mục 1.
*   Lưu các tệp tin này với đúng tên file gốc và đặt vào thư mục **`Input/`** (ghi đè lên file cũ):
    *   `Input/application_train.csv` (chứa dữ liệu mới đã có nhãn trả nợ thực tế).
    *   `Input/bureau.csv`
    *   `Input/previous_application.csv`
    *   `Input/installments_payments.csv`

### BƯỚC 2: Chạy tích hợp và làm sạch dữ liệu
*   Thực thi lệnh sau trên terminal của bạn:
    ```bash
    python step1_2_data_processing.py
    ```
*   **Kết quả**: Hệ thống tự động xử lý file `Input/application_train.csv` cùng các file phụ và tạo ra file dữ liệu học tập sạch tích hợp mới tại **`data_output/cleaned_data.csv`**.

### BƯỚC 3: Huấn luyện lại mô hình
*   Thực thi lệnh sau trên terminal của bạn:
    ```bash
    python step3_4_5_pipeline.py
    ```
*   **Kết quả**: Mô hình được huấn luyện lại trên dữ liệu mới. Nhật ký so sánh hiệu năng, kiểm định t-test và các biểu đồ đặc trưng mới nhất sẽ được tự động sinh ra lần lượt tại thư mục **`results/`** và **`plots/`** với mã thời gian mới.

### BƯỚC 4: Chạy dự báo thực tế cho khách hàng mới
*   Thực thi lệnh sau trên terminal của bạn:
    ```bash
    python predict_new_customers.py
    ```
*   **Kết quả**: Chương trình sẽ nạp mô hình vừa được cập nhật và đưa ra quyết định duyệt vay chính xác nhất cho tệp khách hàng mới theo xu hướng rủi ro hiện tại.
