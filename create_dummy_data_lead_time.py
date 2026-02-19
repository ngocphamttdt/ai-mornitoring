"""
Dataset cho dự báo CRITICAL có lead time ~30-60 phút.
Lịch sử kết thúc bằng ramp tăng nhưng CHƯA vượt ngưỡng 80%,
để Prophet dự báo "sẽ vượt trong 30-60 phút tới" → hệ thống kịp take action.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

num_days = 10
frequency = '15min'
end_date = datetime.now()
start_date = end_date - timedelta(days=num_days)
instances = ['Web-Server-01', 'DB-Server-01']

metric_configs = {
    'Web-Server-01': {'mem': 8, 'disk': 50},
    'DB-Server-01': {'mem': 64, 'disk': 500},
}

# Ramp cuối ~68% → Prophet dự báo vượt 80% sau ~60-90 phút (lead time đủ take action)

data_list = []
date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
n = len(date_range)

for instance in instances:
    hour_effect = np.sin(2 * np.pi * (np.asarray(date_range.hour) - 2) / 24)
    base_values = np.asarray(40 + 20 * hour_effect + np.random.normal(0, 3, n), dtype=float).copy()

    if instance == 'Web-Server-01':
        # Xu hướng tăng dần
        base_values += np.linspace(0, 18, n)
        # Ramp cuối: tăng đều nhưng kết thúc ở ~68% (xa 80%) → Prophet dự báo vượt 80% sau 5-6 bước ≈ 75-90 phút
        n_tail = 10
        end_val = 68.0
        base_values[-n_tail:] = np.linspace(end_val - 10, end_val, n_tail) + np.random.normal(0, 0.5, n_tail)
        base_values[-n_tail:] = np.clip(base_values[-n_tail:], 0, 75)  # chắc chắn dưới 80%

    if instance == 'DB-Server-01':
        base_values = 10 + 5 * hour_effect + np.random.normal(0, 2, n)

    for i, ts in enumerate(date_range):
        val = max(0, min(100, base_values[i]))
        data_list.append({
            'timestamp': ts,
            'instance_id': instance,
            'metric_label': 'CPU_usage',
            'value': round(val, 2),
            'memory_gb': metric_configs[instance]['mem'],
            'storage_gb': metric_configs[instance]['disk'],
        })

df = pd.DataFrame(data_list)
out_path = 'dataset/historical_server_logs_10days_cri_leadtime.csv'
df.to_csv(out_path, index=False)

print(f"✅ Đã tạo {out_path}")
print(f"   Web-Server-01: ramp cuối ~68%, Prophet dự báo vượt 80% trong ~60-90 phút tới (lead time đủ để take action).")
