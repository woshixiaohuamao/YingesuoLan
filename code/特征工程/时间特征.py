import pandas as pd

file_path = r'D:\Aa中工互联\工作安排\英格索兰\data\特征工程\0瞬时流量.xlsx'
df = pd.read_excel(file_path)

df['time'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M:%S')

# 提取年份
df['year'] = df['time'].dt.year
# 提取月份
df['month'] = df['time'].dt.month
# 提取一个月中的第几天
df['day'] = df['time'].dt.day
# 提取星期几（0=星期一，6=星期日）
df['weekday'] = df['time'].dt.weekday
# 提取一天中的小时
df['hour'] = df['time'].dt.hour
# 提取分钟
df['minute'] = df['time'].dt.minute
# 是否是周末（可以根据需求自定义）
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
# 提取季度
df['quarter'] = df['time'].dt.quarter

df.to_excel('时间特征_每分钟数据.xlsx')