import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. CHUẨN BỊ DỮ LIỆU (HISTORICAL DATA) ---
np.random.seed(42)
start_time = datetime.now() - timedelta(days=7)
periods = 1008 # 7 ngày * 24h * 6 (mỗi 10 phút 1 lần)
time_index = [start_time + timedelta(minutes=10*i) for i in range(periods)]

# Giả lập 2 loại dữ liệu: High Usage (Risk) và Low Usage (Opportunity)
# Server 1: Xu hướng tăng (Risk)
val_risk = np.linspace(40, 85, periods) + np.random.normal(0, 2, periods)
# Server 2: Xu hướng thấp (Opportunity)
val_opp = np.linspace(20, 15, periods) + np.random.normal(0, 1, periods)

df_hist = pd.DataFrame({'ds': time_index, 'y': val_risk}) # Định dạng ds, y cho Prophet

# --- 2. TRAINING & PREDICTION ---
# A. Linear Regression
df_hist['minutes'] = (df_hist['ds'] - df_hist['ds'].min()).dt.total_seconds() / 60
lr_model = LinearRegression().fit(df_hist[['minutes']].values, df_hist['y'].values)
future_mins = np.arange(df_hist['minutes'].max() + 10, df_hist['minutes'].max() + 210, 10).reshape(-1, 1)
lr_pred = lr_model.predict(future_mins)

# B. Prophet [cite: 11]
m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
m.fit(df_hist)
future = m.make_future_dataframe(periods=20, freq='10T') # Dự báo 200 phút tiếp theo
forecast = m.predict(future)

# --- 3. LOGIC PHÂN LOẠI RỦI RO (RISK)  ---
threshold = 90
offset = 999
for i, val in enumerate(forecast['yhat'].iloc[-20:]):
    if val >= threshold:
        offset = (i + 1) * 10
        break

risk_status = "Normal"
if offset < 30: risk_status = "CRITICAL"
elif 30 <= offset <= 50: risk_status = "MAJOR"
elif offset > 90 and offset != 999: risk_status = "MINOR"

# --- 4. LOGIC TÍNH TOÁN CƠ HỘI (OPPORTUNITY) [cite: 22, 31, 32] ---
# Giả sử: Memory = 4GB, Storage = 10GB. Công thức: (Mem + Storage) * 30 ngày
memory_cost = 4 
storage_cost = 0.5
monthly_saving = (memory_cost + storage_cost) * 30 # 

opp_status = "Low"
if val_opp.mean() < 20: # Nếu usage thấp
    if monthly_saving > 100: # 
        opp_status = "HIGH"
    elif monthly_saving > 50:
        opp_status = "MEDIUM"

# --- 5. VISUALIZATION [cite: 6] ---
plt.figure(figsize=(12, 6))
plt.plot(df_hist['ds'], df_hist['y'], label='Historical Data (Tier 3)', color='blue')
plt.plot(forecast['ds'].iloc[-20:], forecast['yhat'].iloc[-20:], label='Prophet Prediction', color='red', linestyle='--')
plt.axhline(y=90, color='orange', linestyle=':', label='Risk Threshold (90%)')
plt.title(f"Predictive Analytics\nRisk: {risk_status} (Offset: {offset}m) | Opportunity: {opp_status} (${monthly_saving})")
plt.legend()
plt.savefig('comprehensive_analysis.png')

print(f"Phân loại rủi ro: {risk_status}")
print(f"Khả năng tiết kiệm: {monthly_saving} USD ({opp_status} Opportunity)")