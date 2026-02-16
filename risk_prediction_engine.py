import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 1. Chuẩn bị dữ liệu huấn luyện (Historical-data)
# Giả lập dữ liệu CPU tăng dần trong 7 ngày qua
np.random.seed(42)
start_time = datetime.now() - timedelta(days=7)
periods = 1000 # số điểm dữ liệu
time_index = [start_time + timedelta(minutes=10*i) for i in range(periods)]
values = np.linspace(30, 85, periods) + np.random.normal(0, 3, periods) # Xu hướng tăng

df = pd.DataFrame({'timestamp': time_index, 'value': values})

# 2. Huấn luyện mô hình Linear Regression 
# Chuyển thời gian thành dạng số để máy học (số phút kể từ điểm bắt đầu)
df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
X = df[['minutes']].values
y = df['value'].values

model = LinearRegression()
model.fit(X, y) # Bắt đầu Training [cite: 42]

# 3. Dự báo tương lai (Prediction) [cite: 45]
# Dự báo cho 200 phút tiếp theo
future_mins = np.arange(df['minutes'].max(), df['minutes'].max() + 200, 10).reshape(-1, 1)
predictions = model.predict(future_mins)
future_times = [df['timestamp'].max() + timedelta(minutes=10*i) for i in range(len(predictions))]

# 4. Đánh giá rủi ro dựa trên Offset (Khoảng cách đến ngưỡng 90%) [cite: 58-60]
threshold = 90
offset = 999 # mặc định là rất xa
for i, pred in enumerate(predictions):
    if pred >= threshold:
        offset = i * 10 # Tính toán offset (phút) 
        break

# Phân loại rủi ro theo tài liệu [cite: 58, 59, 60]
risk_status = "Normal"
if offset < 30: risk_status = "CRITICAL"
elif 30 <= offset <= 50: risk_status = "MAJOR"
elif offset > 90: risk_status = "MINOR"

# 5. Visualize (Trực quan hóa dữ liệu từ KV Store/Cache) [cite: 38]
plt.plot(df['timestamp'], df['value'], label='Historical Data', color='blue')
plt.plot(future_times, predictions, label='Prediction Trend', color='red', linestyle='--')
plt.axhline(y=90, color='orange', label='Critical Threshold (90%)')
plt.title(f"Risk Assessment: {risk_status} (Offset: {offset} mins)")
plt.legend()
plt.savefig('server_risk_visualization.png')