import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Cấu hình thông số
num_days = 10
frequency = '15min'  # Thu thập mỗi 15 phút (tương ứng cron job chạy định kỳ)
start_date = datetime(2026, 2, 6)
instances = ['Web-Server-01', 'DB-Server-01']

# Cấu hình ngưỡng cho từng loại metric (Logical-dist) [cite: 7, 8]
metric_configs = {
    'CPU': {'min': 0, 'max': 100, 'unit': '%', 'mem': 8, 'disk': 50},
    'RAM': {'min': 0, 'max': 100, 'unit': '%', 'mem': 16, 'disk': 100}
}

data_list = []

# 2. Tạo dữ liệu chuỗi thời gian (Time-series) [cite: 20]
date_range = pd.date_range(start=start_date, periods=num_days * 24 * 4, freq=frequency)

for instance in instances:
    for metric, config in metric_configs.items():
        # Tạo chu kỳ ngày đêm (Sine wave)
        # Cao điểm vào buổi trưa (12h), thấp điểm vào ban đêm (2h sáng)
        hour_effect = np.sin(2 * np.pi * (date_range.hour - 2) / 24)
        
        # Giá trị cơ sở + Nhiễu ngẫu nhiên
        base_values = 40 + 20 * hour_effect + np.random.normal(0, 5, len(date_range))
        
        # Giả lập xu hướng tăng dần để tạo "Risk" cho Web-Server [cite: 25]
        if instance == 'Web-Server-01' and metric == 'CPU':
            base_values += np.linspace(0, 35, len(date_range)) 
            
        # Giả lập xu hướng cực thấp để tạo "Opportunity" cho DB-Server [cite: 21]
        if instance == 'DB-Server-01':
            base_values -= 25

        for i, ts in enumerate(date_range):
            val = max(0, min(100, base_values[i]))
            
            data_list.append({
                'timestamp': ts,
                'instance_id': instance,
                'metric_label': f"{metric}_usage", # [cite: 8]
                'value': round(val, 2),            # Historical-data [cite: 9]
                'min_range': config['min'],        # Logical-dist min 
                'max_range': config['max'],        # Logical-dist max 
                'memory_gb': config['mem'],        # Phục vụ tính Cost Saving 
                'storage_gb': config['disk']       # Phục vụ tính Cost Saving 
            })

# 3. Xuất file CSV (Tập dữ liệu thô Tier 3) [cite: 4, 12]
df = pd.DataFrame(data_list)
df.to_csv('historical_server_logs_10days.csv', index=False)

print(f"Hoàn thành! Đã tạo {len(df)} bản ghi vào file 'historical_server_logs_10days.csv'")
print(df.head())