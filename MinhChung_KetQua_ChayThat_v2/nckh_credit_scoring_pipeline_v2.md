# NGHIÊN CỨU HỆ THỐNG CHẤM ĐIỂM TÍN DỤNG & DỰ BÁO RỦI RO BÙNG NỢ (V2)
## Credit Scoring & Default Risk Prediction Pipeline - Vietnamese Logical Dataset

### ABSTRACT
This research develops an advanced credit scoring pipeline using alternative behavioral data to predict loan default risk. By integrating traditional application profiles with bureau history and payment patterns, we engineered 24 robust features. We evaluated four state-of-the-art algorithms—Random Forest, LightGBM, XGBoost, and CatBoost—using 3-fold stratified cross-validation on a dataset of **15,000 loan records**. Our experimental results demonstrate significant predictive power, **establishing CatBoost as the optimal base learner** with a mean AUC-ROC of approximately 0.84. Furthermore, we implemented a cost-sensitive learning framework that reduced the Expected Financial Loss (EFL) by **910.00 million VND**, showcasing the model's practical utility in banking risk management.

---

### 1. GIỚI THIỆU (INTRODUCTION)
Hệ thống này tập trung vào việc giải quyết bài toán mất cân bằng dữ liệu trong tín dụng và xóa bỏ các rào cản về dữ liệu thô không đồng nhất. Quy trình bao gồm:
1. **Tích hợp dữ liệu sạch**: Sử dụng bộ dữ liệu giả lập 18,000 mẫu (15,000 train / 3,000 test) với logic tài chính Việt Nam.
2. **Thiết kế đặc trưng (Feature Engineering)**: Tập trung vào các chỉ số DPD (Days Past Due), DIR (Debt-to-Income) và điểm tín dụng chuẩn hóa (300-850).
3. **So sánh thuật toán**: Chứng minh tính vượt trội của **CatBoost** trong việc xử lý các biến phân loại và dữ liệu nhiễu.
4. **Tối ưu hóa kinh tế**: Sử dụng hàm tổn thất EFL để đưa ra quyết định duyệt vay sát với thực tế kinh doanh.

### 2. THỰC THI MÃ NGUỒN (PIPELINE CODE)

```python
# BƯỚC 1 & 2: TIỀN XỬ LÝ VÀ FEATURE ENGINEERING
# (Xem chi tiết tại step1_2_data_processing_v2.py)
# Dữ liệu đầu vào: Input2/ (15,000 bản ghi huấn luyện)
# Dữ liệu đầu ra: data_output_v2/cleaned_data2.csv
```

### 3. KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)

#### 3.1. So sánh hiệu năng thuật toán
Dựa trên kết quả chạy thực tế, **CatBoost** đạt hiệu năng cao nhất với độ ổn định vượt trội.

| Thuật Toán | AUC-ROC (Mean) | Hệ Số Gini | Chỉ Số KS | Brier Score |
|---|---|---|---|---|
| Random Forest | 0.8303 | 0.6605 | 0.5834 | 0.0368 |
| LightGBM | 0.8298 | 0.6595 | 0.5802 | 0.0365 |
| XGBoost | 0.8330 | 0.6661 | 0.5898 | 0.0362 |
| **CatBoost (Best)** | **0.8429** | **0.6858** | **0.6029** | **0.0356** |

#### 3.2. Ma trận nhầm lẫn và Hiệu quả kinh tế
Mô hình cải tiến giúp bắt trúng thêm các ca nợ xấu mà không làm gia tăng quá mức tỷ lệ từ chối nhầm khách hàng tốt.

- **Số mẫu thử nghiệm**: 3,000 hồ sơ.
- **Tiết kiệm EFL**: **910.00 triệu VND**.

### 4. GIẢI THÍCH AI (EXPLAINABLE AI - XAI)
Biểu đồ SHAP và Feature Importance chỉ ra rằng **Điểm tín dụng nguồn 1, 2, 3** và **Số ngày trễ hạn trung bình (INST_PAY_DELAY_MEAN)** là những biến có tác động mạnh nhất đến rủi ro.

---
### 5. CÔNG THỨC TOÁN HỌC ĐÓNG GÓP
1. **Expected Financial Loss (EFL)**:
   $$EFL = (FN \times 100M) + (FP \times 10M)$$
2. **AUC-ROC Probabilistic Interpretation**:
   $$AUC = P(f(x_{bad}) > f(x_{good}))$$

*Báo cáo này được trích xuất tự động từ hệ thống thực nghiệm V2.*
