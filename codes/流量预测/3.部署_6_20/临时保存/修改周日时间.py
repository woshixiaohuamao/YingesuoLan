import pandas as pd
from datetime import datetime, timedelta


file_path = ''  
df = pd.read_excel(file_path)


print("原始数据行数：", len(df))

points_per_day = 48


num_days = len(df) // points_per_day
print(f"数据总共包含 {num_days} 天（周日）")


today = datetime.today()
days_since_sunday = today.weekday() + 1 
last_sunday = today - timedelta(days=days_since_sunday)

print(f"最近的周日为：{last_sunday.date()}")

new_timestamps = []

for day in range(num_days):
    date_for_day = last_sunday - timedelta(days=7 * day) 
    for point in range(points_per_day):
        new_time = date_for_day + timedelta(minutes=30 * point)
        new_timestamps.append(new_time)


new_timestamps = new_timestamps[::-1]


assert len(new_timestamps) == len(df), "新时间戳数量与原数据行数不一致！"
df['timestamp'] = new_timestamps
output_path = 'modified_file.xlsx'  
df.to_excel(output_path, index=False)

print(f"修改后的文件已保存为：{output_path}")
