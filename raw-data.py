import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Cấu hình thông số cơ bản
num_days = 7
frequency = '1min'  # 1 minute (tương ứng mỗi phút chạy cron job) 
start_date = datetime(2026, 2, 7)
instances = ['server-01', 'server-02']
metrics = ['CPU', 'RAM']

# 2. Tạo khung thời gian
date_range = pd.date_range(start=start_date, periods=num_days*24*60, freq=frequency)

data_list = []

for instance in instances:
    for metric in metrics:
        # Tạo dữ liệu ngẫu nhiên có tính xu hướng (để mô hình Prophet/Linear Regression có thể học)
        base_usage = np.random.uniform(20, 50, size=len(date_range))
        
        # Giả lập một đợt High Usage (Risk) - Tăng dần lên trên 90% [cite: 25]
        base_usage[500:1000] += np.linspace(0, 45, 500) 
        
        # Giả lập một đợt Low Usage (Opportunity) - Giảm xuống dưới 20% [cite: 21]
        base_usage[2000:2500] -= np.linspace(0, 15, 500)

        for i, ts in enumerate(date_range):
            usage = max(0, min(100, base_usage[i])) # Đảm bảo trong khoảng 0-100%
            
            data_list.append({
                'timestamp': ts,
                'instance_id': instance,
                'metric_label': metric, # [cite: 8]
                'value': round(usage, 2), # Historical-data [cite: 9]
                'min_range': 0, # 
                'max_range': 100 # 
            })

# 3. Tạo DataFrame và lưu file
df = pd.DataFrame(data_list)
df.to_csv('historical_server_logs.csv', index=False)

print("Đã tạo thành công file 'historical_server_logs.csv' với", len(df), "dòng dữ liệu.")
print(df.head())