# Khung Nghiên Cứu và Triển Khai Mô Hình Chấm Điểm Tín Dụng

---

## BƯỚC 1: KHỞI TẠO & NẠP DỮ LIỆU ĐA CHIỀU (DATA PREPARATION)

- **Việc cần làm:** Chạy ô code nạp dữ liệu truyền thống (`application_train`) kết hợp với dữ liệu hành vi thay thế (`bureau`).
- **Mục tiêu kinh tế:** Chứng minh việc kết hợp đa nguồn dữ liệu giúp đánh giá khách hàng toàn diện hơn.
- **Output cần thu hoạch:**
  - Con số kích thước tập dữ liệu sau khi gộp (Số hàng, Số cột $\rightarrow$ Biến số tăng lên bao nhiêu).
  - Tỷ lệ mất cân bằng nợ xấu thực tế trong lịch sử (Bao nhiêu % `TARGET = 0` và bao nhiêu % `TARGET = 1`).

---

## BƯỚC 2: TIỀN XỬ LÝ & MÃ HÓA (PREPROCESSING)

- **Việc cần làm:** Chạy code chuyển đổi toàn bộ các cột chữ (Giới tính, Học vấn, Loại tài sản, Ngành nghề) thành số thông qua `LabelEncoder` để máy tính có thể tính toán toán học.
- **Output cần thu hoạch:**
  - Một bộ dữ liệu sạch 100% dạng số, đã được chia tách thành 2 tập: Train (80% để học) và Validation (20% để kiểm tra độ chính xác).

---

## BƯỚC 3: HUẤN LUYỆN & SO SÁNH MÔ HÌNH (MODEL COMPARISON)

- **Việc cần làm:** Chạy song song 2 thuật toán mạnh nhất hiện nay cho dữ liệu bảng là LightGBM và XGBoost trên cùng một tập dữ liệu đã chuẩn bị.
- **Output cần thu hoạch:** Bạn lập một bảng so sánh trong bài nghiên cứu với các thông số sau:
  - LightGBM: Điểm AUC-ROC? Hệ số Gini?
  - XGBoost: Điểm AUC-ROC? Hệ số Gini?
  - _(Công thức tính Gini rất đơn giản: $Gini = 2 \times AUC - 1$)_

---

## BƯỚC 4: TRIỂN KHAI ĐIỂM CẢI TIẾN (MODEL IMPROVEMENT)

- **Việc cần làm:** Áp dụng giải pháp xử lý mất cân bằng dữ liệu bằng cách thêm trọng số `scale_pos_weight = 11.5` vào mô hình tốt nhất ở Bước 3.
- **Output cần thu hoạch:**
  - Điểm AUC và Gini sau khi cải tiến.
  - Bảng ma trận nhầm lẫn (Confusion Matrix) cho thấy mô hình mới đã tăng số lượng bắt trúng các ca bùng nợ thực tế lên bao nhiêu ca so với mô hình cũ.

---

## BƯỚC 5: TRÍCH XUẤT BIẾN SỐ TÀI CHÍNH QUAN TRỌNG (FEATURE IMPORTANCE)

- **Việc cần làm:** Xuất biểu đồ top các chỉ số tài chính và hành vi tác động mạnh nhất đến rủi ro bùng nợ của khách hàng.
- **Output cần thu hoạch:**
  - File ảnh biểu đồ `feature_importance.png` (Top 15 biến).
  - Danh sách tên các biến số đó để đưa vào phần luận văn nhằm giải thích dưới góc nhìn kinh tế (Ví dụ: `EXT_SOURCE` - điểm uy tín từ tổ chức bên ngoài, hay `DAYS_BIRTH` - độ tuổi ảnh hưởng thế nào đến rủi ro).
