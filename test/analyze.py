import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

entropy = []

# 打开CSV文件
with open('download/output_1GB.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx != 0 and len(row) == 6:
            entropy.append(float(row[3]))


data = pd.DataFrame({'x': np.arange(len(entropy)), 'y': np.flip(np.sort(entropy))})
sampled_data = data.sample(frac=1)


plt.plot(sampled_data['x'],sampled_data['y'], 'r.', markersize=1, alpha=0.1)
plt.show()