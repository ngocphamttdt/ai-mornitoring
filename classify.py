def classify_monitoring_data(offset_minutes, cost_saving_amount=0, usage_type='high'):
    """
    Phân loại trạng thái hệ thống dựa trên tài liệu quản lý rủi ro.
    """
    
    # 1. Logic phân loại Rủi ro (High Usage / Risk) [cite: 25]
    if usage_type == 'high':
        if offset_minutes < 30:
            return "Critical"  # Offset < 30 
        elif 30 <= offset_minutes <= 50:
            return "Major"     # 30 < offset < 50 
        elif offset_minutes > 90:
            return "Minor"     # offset > 90 
        else:
            return "Normal"

    # 2. Logic phân loại Cơ hội tiết kiệm (Low Usage / Opportunity) [cite: 18, 21]
    elif usage_type == 'low':
        if cost_saving_amount > 100:
            return "High Opportunity"   # Tiết kiệm > 100 USD 
        elif 50 <= cost_saving_amount <= 100:
            return "Medium Opportunity" # [cite: 23]
        else:
            return "Low Opportunity"    # [cite: 24]

    return "Normal"

# Giả lập dữ liệu dự báo (Prediction data)
prediction_results = [
    {"instance": "server-01", "offset": 25, "usage": "high", "saving": 0},
    {"instance": "server-02", "offset": 100, "usage": "low", "saving": 150},
    {"instance": "server-03", "offset": 45, "usage": "high", "saving": 0}
]

print(f"{'Instance':<12} | {'Status':<15} | {'Action Type'}")
print("-" * 45)

for res in prediction_results:
    status = classify_monitoring_data(res['offset'], res['saving'], res['usage'])
    print(f"{res['instance']:<12} | {status:<15} | {res['usage'].upper()}")