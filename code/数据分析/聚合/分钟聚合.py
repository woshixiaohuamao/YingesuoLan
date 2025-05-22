import pandas as pd

file_path = r'D:\Aa中工互联\工作安排\英格索兰\data\预测\周预测.xlsx'
df = pd.read_excel(file_path)

# 数据预处理
df['时间'] = pd.to_datetime(df['时间'])
df.set_index('时间', inplace=True)
df = df.sort_index()

# 计算每分钟平均压力
weekly_pressure = df.resample('min')[['DQ200系统压力', 'AVS系统压力']].mean()

# 计算每分钟总流量（周日值 - 周一值）
weekly_flow = df.resample('min').agg(
    DQ200总流量=('DQ200累积流量', lambda x: x.iloc[-1] - x.iloc[0]),
    AVS总流量=('AVS累积流量', lambda x: x.iloc[-1] - x.iloc[0])
)

# 合并结果
weekly_data = weekly_pressure.join(weekly_flow)

# 保存结果
weekly_data.to_excel('每分钟数据.xlsx')
print('数据保存成功！')