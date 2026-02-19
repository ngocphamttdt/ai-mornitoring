import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Tạo 21 ngày (3 tuần) để Prophet học chu kỳ tuần cực chuẩn
end_date = datetime(2026, 2, 18, 23, 0)
num_days = 21 
timestamps = pd.date_range(end=end_date, periods=num_days * 96, freq='15min')
data_list = []

for ts in timestamps:
    # Nền tảng 40%, có chút nhiễu nhẹ cho thật
    val = 40 + np.random.normal(0, 1)
    
    # Đỉnh Sale: Thứ 2 (0) và Thứ 5 (3) từ 10h-16h
    if ts.weekday() in [0, 3] and (10 <= ts.hour <= 16):
        # Tạo đỉnh vòm mượt mà thay vì nhảy vọt
        peak_shape = np.sin(np.pi * (ts.hour - 10) / 6)
        val += 50 * peak_shape 
        
    data_list.append({
        'timestamp': ts, 'instance_id': 'Web-Server-01',
        'metric_label': 'CPU_usage', 'value': round(val, 2),
        'memory_gb': 16, 'storage_gb': 100
    })

pd.DataFrame(data_list).to_csv('historical_long.csv', index=False)
print("✅ Đã tạo xong dữ liệu 3 tuần với chu kỳ Sale Thứ 2 & Thứ 5 cực rõ.")