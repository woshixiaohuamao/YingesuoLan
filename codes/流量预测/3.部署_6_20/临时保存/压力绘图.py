import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r''  
df = pd.read_excel(file_path)


df['timestamp'] = pd.to_datetime(df['timestamp'])

df.sort_values('timestamp', inplace=True)


sns.set_style("whitegrid")


plt.figure(figsize=(10, 6))
sns.lineplot(x='timestamp', y='forecast', data=df)
plt.title('Time Series Plot of Forecast')
plt.xlabel('Timestamp')
plt.ylabel('Forecast Value')

plt.ylim(0, 8) 
plt.yticks(range(0, 8, 1))  
plt.show()