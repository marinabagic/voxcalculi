import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

dirs = [r"D:\diplomski\datasets\mathematics_dataset-v1.0\train-easy",
        r"D:\diplomski\datasets\mathematics_dataset-v1.0\train-medium",
        r"D:\diplomski\datasets\mathematics_dataset-v1.0\train-hard"]

data = {'Category': [], 'Count': [], 'Training data': []}

for dir in dirs:

    files = [f for f in os.listdir(dir) if f.endswith('.txt')]

    for file in files:
        if all(os.path.exists(os.path.join(d, file)) for d in dirs):
            with open(os.path.join(dir, file)) as f:
                count = sum(1 for line in f)

            category = file.split('__')[0]

            data['Category'].append(category)
            data['Count'].append(count)
            data['Training data'].append(dir.split(os.sep)[-1])

df = pd.DataFrame(data)

df_grouped = df.groupby(['Category', 'Training data'], as_index=False).sum()

max_counts = df_grouped.groupby('Category')['Count'].max().values

plt.figure(figsize=(10,6))
sns.barplot(x='Category', y='Count', hue='Training data', data=df_grouped)
plt.ylabel('Number of Rows')

plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.yticks(list(set(max_counts)))

plt.tight_layout(pad=1.0)
plt.show()