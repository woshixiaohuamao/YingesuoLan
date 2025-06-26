import pandas as pd
import matplotlib.pyplot as plt


file_name = r''
with open(file_name, 'r', encoding='utf-8') as f:
    data = pd.read_json(f)


print(data.head())


data.to_excel(r'', index=False)
print('保存成功！')