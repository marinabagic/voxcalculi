import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

sns.set_theme()

dirs = [r"D:\diplomski\datasets\mathematics_dataset-v1.0\interpolate"]

data = {'Category': [], 'Count': [], 'Training data': []}

for dir in dirs:
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]

    for file in files:
        if all(os.path.exists(os.path.join(d, file)) for d in dirs):
            with open(os.path.join(dir, file)) as f:
                count = sum(1 for line in f)

            data['Category'].append(file[:-4])
            data['Count'].append(count)
            data['Training data'].append(dir.split(os.sep)[-1])

df = pd.DataFrame(data)

plt.figure(figsize=(6,30))
sns.barplot(x='Count', y='Category', data=df)
plt.ylabel('Modules')
plt.xticks([df['Count'].max()])
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout(pad=5.0)
plt.show()