# Hướng Dẫn Định Dạng Dữ Liệu & Cập Nhật Hệ Thống (Dành Cho NHTM)

Tài liệu này hướng dẫn chi tiết cách phòng Quản trị rủi ro và phòng Công nghệ thông tin của Ngân hàng chuẩn bị cấu trúc tệp tin dữ liệu đầu vào (Data Schema) và các bước thực hiện để cập nhật mô hình hoặc dự đoán khách hàng mới.

---

## 1. ĐỊNH DẠNG CÁC TỆP TIN DỮ LIỆU ĐẦU VÀO (INPUT SCHEMA)

Để hệ thống hoạt động chính xác không bị lỗi cấu trúc (schema error), các tệp tin CSV đặt trong thư mục **`Input/`** phải tuân thủ nghiêm ngặt định dạng cột sau đây:

### 1.1. Tệp tin hồ sơ khách hàng (`application_train.csv` & `application_test.csv`)

- **Định dạng**: File CSV, sử dụng dấu phẩy `,` làm dấu phân cách.
- **Sự khác biệt**:
  - `application_train.csv` (dùng để dạy học): Bắt buộc phải có cột **`TARGET`** (nhãn rủi ro: `0` là trả tốt, `1` là nợ xấu).
  - `application_test.csv` (dùng để dự đoán khách hàng mới): **Không có cột `TARGET`**.
- **Các trường thông tin cốt lõi bắt buộc phải có**:

| Tên Cột              | Kiểu Dữ Liệu      | Ý Nghĩa Nghiệp Vụ                                                 | Ví Dụ               |
| :------------------- | :---------------- | :---------------------------------------------------------------- | :------------------ |
| `SK_ID_CURR`         | Số nguyên (`int`) | Mã định danh duy nhất của hồ sơ khách hàng                        | `100002`            |
| `AMT_INCOME_TOTAL`   | Số thực (`float`) | Tổng thu nhập hàng năm của khách hàng                             | `135000.0`          |
| `AMT_CREDIT`         | Số thực (`float`) | Số tiền khách hàng đăng ký vay                                    | `450000.0`          |
| `AMT_ANNUITY`        | Số thực (`float`) | Số tiền khách hàng phải trả định kỳ (gốc + lãi)                   | `25000.0`           |
| `DAYS_BIRTH`         | Số nguyên (`int`) | Độ tuổi khách hàng (tính bằng số ngày âm tính ngược từ hiện tại)  | `-12000` (~33 tuổi) |
| `DAYS_EMPLOYED`      | Số nguyên (`int`) | Thâm niên làm việc tại đơn vị hiện tại (số ngày âm)               | `-1500` (~4 năm)    |
| `EXT_SOURCE_1, 2, 3` | Số thực (`float`) | Điểm tín dụng từ các đối tác hoặc CIC bên thứ ba (từ 0.0 đến 1.0) | `0.556`, `0.623`    |

---

### 1.2. Tệp tin lịch sử tín dụng ngoài hệ thống (`bureau.csv`)

- **Vai trò**: Cung cấp thông tin khách hàng đang nợ bao nhiêu khoản ở nơi khác.
- **Các trường bắt buộc**:
  - `SK_ID_CURR` (Số nguyên): Để liên kết với mã khách hàng.
  - `DAYS_CREDIT` (Số nguyên): Số ngày kể từ khi đăng ký khoản vay ngoài.
  - `SK_ID_BUREAU` (Số nguyên): Mã định danh khoản vay ngoài (dùng để đếm số lượng khoản vay).

---

### 1.3. Tệp tin lịch sử đơn vay cũ trong hệ thống (`previous_application.csv`)

- **Vai trò**: Cung cấp thông tin lịch sử giao dịch của khách hàng tại chính ngân hàng đó.
- **Các trường bắt buộc**:
  - `SK_ID_CURR` (Số nguyên): Liên kết với mã khách hàng.
  - `SK_ID_PREV` (Số nguyên): Mã khoản vay cũ (dùng để đếm số lượng đơn vay quá khứ).
  - `AMT_CREDIT` (Số thực): Số tiền vay được phê duyệt của các khoản vay cũ.
  - `NAME_CONTRACT_STATUS` (Chữ): Trạng thái đơn vay cũ (ví dụ: `Approved`, `Refused`, `Canceled`).

---

### 1.4. Tệp tin lịch sử đóng tiền định kỳ (`installments_payments.csv`)

- **Vai trò**: Cung cấp thói quen trả nợ (trả trễ hạn, trả thiếu tiền).
- **Các trường bắt buộc**:
  - `SK_ID_CURR` (Số nguyên): Liên kết với mã khách hàng.
  - `DAYS_INSTALMENT` (Số nguyên): Ngày đến hạn thanh toán theo hợp đồng.
  - `DAYS_ENTRY_PAYMENT` (Số nguyên): Ngày khách hàng thực tế đóng tiền.
  - `AMT_INSTALMENT` (Số thực): Số tiền bắt buộc phải đóng theo đợt.
  - `AMT_PAYMENT` (Số thực): Số tiền khách hàng thực tế đóng.

---

## 2. QUY TRÌNH 4 BƯỚC ĐỂ CẬP NHẬT MÔ HÌNH (MODEL RETRAINING)

Khi ngân hàng có dữ liệu của chu kỳ kinh doanh mới (ví dụ sau 6 tháng) và muốn cập nhật mô hình để tăng độ chính xác:

### BƯỚC 1: Chuẩn bị và đặt file mới vào đúng vị trí

- Trích xuất dữ liệu mới từ Database của ngân hàng theo đúng cấu trúc cột ở mục 1.
- Lưu các file này với đúng tên file gốc và đặt vào thư mục **`Input/`** (ghi đè lên file cũ):
  - `Input/application_train.csv` (chứa dữ liệu mới đã có kết quả trả nợ thực tế).
  - `Input/bureau.csv`
  - `Input/previous_application.csv`
  - `Input/installments_payments.csv`

### BƯỚC 2: Xóa các kết quả chạy cũ để tránh nhầm lẫn (Khuyên dùng)

- Xóa sạch các file cũ trong thư mục **`results/`** và **`plots/`**.

### BƯỚC 3: Chạy tích hợp và làm sạch dữ liệu

- Chạy lệnh sau trên terminal:
  ```bash
  python step1_2_data_processing.py
  ```
- _Kết quả_: Hệ thống tự động xử lý file `Input/application_train.csv` cùng các file phụ và tạo ra file dữ liệu học tập sạch tích hợp mới tại **`data_output/cleaned_data.csv`**.

### BƯỚC 4: Huấn luyện lại bộ não mô hình

- Chạy lệnh sau trên terminal:
  ```bash
  python step3_4_5_pipeline.py
  ```
- _Kết quả_: Mô hình được huấn luyện lại trên dữ liệu mới để học thêm các hành vi bùng nợ mới phát sinh. Kết quả so sánh độ chính xác và biểu đồ đặc trưng mới nhất sẽ được sinh ra lần lượt tại thư mục **`results/`** và **`plots/`** với mã thời gian mới.
