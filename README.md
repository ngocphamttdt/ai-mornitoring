# Giải thích Notebook `training-more.ipynb`

Tài liệu này giải thích cấu trúc code, **Linear Regression** và **Prophet** trong notebook dự báo tải CPU và quản trị rủi ro server.

---

## 1. Tổng quan

Notebook thực hiện:

- **Nạp dữ liệu** lịch sử CPU (10 ngày) cho nhiều server.
- **Dự báo** tải CPU 24h tới bằng **Prophet** và **Linear Regression**.
- **Phân loại rủi ro**: CRITICAL / MAJOR / MINOR / NORMAL dựa trên ngưỡng 75%.
- **Hành động**: in thông báo quản trị (auto-scaling, cảnh báo, cơ hội tiết kiệm chi phí).
- **Trực quan hóa**: lịch sử, dự báo Prophet, xu hướng LR, và thành phần chu kỳ.

---

## 2. Cấu trúc code theo từng block

### Block 1: Khai báo thư viện và nạp dữ liệu

- **Thư viện**: `pandas`, `numpy`, `matplotlib`, `sklearn.linear_model.LinearRegression`, `prophet.Prophet`, `datetime`, `warnings`.
- **Dữ liệu**: đọc file `historical_server_logs_10days_cri_leadtime.csv`, cột `timestamp` chuyển sang `datetime`.
- **Danh sách server**: `all_instances = df['instance_id'].unique()` — duyệt từng server trong vòng lặp sau.

### Block 2: Hàm Take Action — `process_server_actions(...)`

Hàm in ra bảng thông báo quản trị cho từng server:

- **Đầu vào**: `instance_id`, `risk_label` (NORMAL/CRITICAL/MAJOR/MINOR), `offset` (phút tới khi vượt ngưỡng), `avg_usage`, `mem_gb`, `storage_gb`, `risk_source` (prophet / lr / recent / fallback).
- **Phần rủi ro**: in trạng thái CRITICAL/MAJOR và hành động (auto-scaling, lập kế hoạch bảo trì).
- **Phần chi phí**: ước tính tiết kiệm dựa trên RAM/disk và gợi ý downsize nếu usage thấp.

### Block 3: Vòng lặp phân tích (ML + phân loại + vẽ đồ thị)

Với mỗi `instance_id`:

1. Lọc và sắp xếp dữ liệu theo thời gian.
2. Chuẩn bị dữ liệu cho Prophet: đổi tên cột `timestamp` → `ds`, `value` → `y`.
3. **Huấn luyện Prophet** và **dự báo 24h** (96 bước 15 phút); clip dự báo trong [0, 100].
4. **Phân loại rủi ro**:
   - Nếu dữ liệu gần đây (8 điểm cuối) đã ≥ 75% → CRITICAL, `risk_source = "recent"`.
   - Nếu Prophet dự báo vượt 75%: theo `offset` (phút tới điểm đầu tiên vượt ngưỡng) → CRITICAL (≤30 phút), MAJOR (≤90 phút), MINOR (>90 phút), `risk_source = "prophet"`.
   - Nếu vẫn NORMAL: dùng **Linear Regression** trên 12 điểm cuối; nếu đường thẳng cắt ngưỡng trong 5–180 phút → CRITICAL (≤45 phút) hoặc MAJOR, `risk_source = "lr"`.
   - Fallback: với Web-Server-01, nếu giá trị cuối ≥ 60% → CRITICAL trong 60 phút, `risk_source = "fallback"`.
5. **Vẽ đồ thị**: lịch sử, dự báo Prophet (và khoảng tin cậy), xu hướng LR (nếu có), ngưỡng 75%, đường “bắt đầu dự báo”.
6. **Biểu đồ thành phần Prophet**: trend, daily/weekly seasonality.
7. Gọi `process_server_actions(...)` với các tham số đã tính.

---

## 3. Linear Regression trong code

### Linear Regression là gì?

**Hồi quy tuyến tính** mô hình quan hệ giữa biến độc lập \(x\) và biến phụ thuộc \(y\) bằng một đường thẳng:

\[
y = \beta_0 + \beta_1 x
\]

- \(\beta_0\): intercept (hệ số chặn).  
- \(\beta_1\): slope (độ dốc).  
Mô hình giả định \(y\) thay đổi theo \(x\) theo dạng tuyến tính; phù hợp để nắm **xu hướng ngắn hạn** (tăng/giảm gần như thẳng).

### Cách dùng trong notebook

- **Dữ liệu**: 12 điểm đo cuối cùng của CPU (`tail = df_target.tail(12)`).  
- **Biến**:  
  - \(x\): chỉ số thời gian (0, 1, 2, ..., 11) — `X = np.arange(len(tail)).reshape(-1, 1)`.  
  - \(y\): giá trị CPU tương ứng — `y = tail['value'].values`.  
- **Huấn luyện**: `lr = LinearRegression().fit(X, y)`.  
- **Dự báo**: ngoại suy 96 bước (15 phút/bước) về phía tương lai:  
  `X_fut = np.arange(len(tail), len(tail) + 96).reshape(-1, 1)`  
  `y_fut = lr.predict(X_fut).clip(0, 100)`.  
- **Mục đích**:  
  - Nếu Prophet báo NORMAL nhưng **xu hướng gần đây tăng mạnh**, LR giúp phát hiện “ramp” — đường thẳng cắt ngưỡng 75% trong 5–180 phút.  
  - Thời điểm cắt ngưỡng: `lr_offset_min = (index_đầu_tiên_vượt_75% + 1) * 15` (phút).  
  - Chỉ áp dụng khi `lr.coef_[0] > 0` (xu hướng tăng).

**Tóm lại**: Linear Regression được dùng như **mô hình bổ sung** để bắt xu hướng tuyến tính ngắn hạn từ 12 điểm gần nhất, tránh bỏ sót trường hợp Prophet chưa kịp phản ứng với ramp nhanh.

---

## 4. Prophet trong code

### Prophet là gì?

**Prophet** (Facebook/Meta) là thư viện dự báo chuỗi thời gian theo thời gian. Mô hình dạng cộng tính:

\[
y = g(t) + s(t) + h(t) + \varepsilon
\]

- **\(g(t)\)**: trend (xu hướng dài hạn, có thể tuyến tính hoặc logistic).  
- **\(s(t)\)**: seasonality (chu kỳ ngày/tuần/năm).  
- **\(h(t)\)**: holiday effects (trong code không dùng).  
- **\(\varepsilon\)**: nhiễu.

Prophet ước lượng từng thành phần bằng fitting; dự báo bằng cách ngoại suy trend và lặp lại seasonality. Ưu điểm: dễ dùng, ổn định với missing data và thay đổi trend, có khoảng tin cậy.

### Cách dùng trong notebook

- **Chuẩn bị dữ liệu**: DataFrame với 2 cột bắt buộc — `ds` (datetime) và `y` (giá trị cần dự báo).  
  `df_p = df_target[['timestamp', 'value']].rename(columns={'timestamp':'ds', 'value':'y'})`.
- **Khởi tạo và fit**:
  - `m = Prophet(daily_seasonality=True, weekly_seasonality=True, seasonality_mode='additive')`.  
  - `m.fit(df_p)`.
- **Dự báo**:
  - `future = m.make_future_dataframe(periods=96, freq='15min')` — thêm 96 bước 15 phút vào tương lai.  
  - `forecast = m.predict(future)`.  
  - Chỉ lấy phần tương lai: `prediction = forecast[forecast['ds'] > last_ts]`.  
- **Clip**: `yhat`, `yhat_lower`, `yhat_upper` được clip trong [0, 100] vì CPU là phần trăm.
- **Phân loại rủi ro**:  
  - `risk_points = prediction[prediction['yhat'] >= 75]`.  
  - Nếu có điểm vượt ngưỡng: `offset = (risk_points['ds'].iloc[0] - last_ts).total_seconds() / 60` (phút tới lần đầu vượt 75%).  
- **Trực quan**:  
  - Vẽ `yhat` (dự báo), `yhat_lower`–`yhat_upper` (khoảng tin cậy).  
  - `m.plot_components(forecast)` để xem trend và chu kỳ ngày/tuần.

**Tóm lại**: Prophet đóng vai trò **mô hình chính** để dự báo CPU 24h tới có tính đến trend và seasonality; kết quả dùng để gán nhãn CRITICAL/MAJOR/MINOR và thời điểm `offset`.

---

## 5. Luồng kết hợp Prophet + Linear Regression

1. **Prophet** dự báo 24h, có seasonality → quyết định rủi ro chính (CRITICAL/MAJOR/MINOR) và `offset` từ dự báo.  
2. Nếu Prophet cho **NORMAL**:  
   - **Linear Regression** trên 12 điểm cuối kiểm tra xem xu hướng thẳng có cắt 75% trong 5–180 phút không.  
   - Nếu có và slope > 0 → ghi đè thành CRITICAL hoặc MAJOR với `risk_source = "lr"`.  
3. **Dữ liệu gần đây** (8 điểm cuối ≥ 75%) → luôn CRITICAL, `risk_source = "recent"`.  
4. **Fallback** (ví dụ Web-Server-01, giá trị cuối ≥ 60%) → CRITICAL 60 phút, `risk_source = "fallback"`.

Như vậy: Prophet xử lý dự báo có chu kỳ; LR bổ sung phát hiện ramp tuyến tính ngắn hạn; hai nguồn kết hợp với quy tắc rõ ràng để ra nhãn và hành động.

---

## 6. Bảng tóm tắt

| Thành phần            | Vai trò trong code |
|-----------------------|--------------------|
| **Linear Regression** | Xu hướng tuyến tính từ 12 điểm cuối; bổ sung khi Prophet NORMAL nhưng ramp nhanh; tính thời điểm cắt ngưỡng 75%. |
| **Prophet**           | Dự báo chính 24h có trend + daily/weekly seasonality; dùng để phân loại CRITICAL/MAJOR/MINOR và `offset`. |
| **Ngưỡng**            | 75% CPU; 30/90 phút để phân biệt CRITICAL/MAJOR (Prophet); 45 phút cho LR. |
| **risk_source**       | `prophet` / `lr` / `recent` / `fallback` — cho biết nguồn quyết định rủi ro. |

---

*Tài liệu giải thích dựa trên notebook `training-more.ipynb`.*
