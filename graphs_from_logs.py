import pandas as pd
import re
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
sns.set_theme()

train_data = []
eval_data = []

with open("C:\\Users\\anteg\\Desktop\\PROJEKTI\\TP-Transformer-master\\trained\\arithmetic\\lr=0.0001_bs=128_h=512_f=2048_nl=6_nh=8_d=0.0_Adam_\\195932581\\output.log", 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if re.match(r'\d+: loss=', line):
        train_data.append(line.strip())
    elif line.startswith('eval:'):
        eval_data.append(line.strip())

train_df = pd.DataFrame()
eval_df = pd.DataFrame()

for data in tqdm(train_data):
    result = re.search(r'(\d+): loss=(\d+\.\d+) acc=(\d+\.\d+) steps', data)
    train_df = train_df.append({
        'Steps': int(result.group(1)),
        'Loss': float(result.group(2)),
        'Acc': float(result.group(3))
    }, ignore_index=True)

for data in tqdm(eval_data):
    result = re.search(r'eval: loss=(\d+\.\d+) acc=(\d+\.\d+) b', data)
    eval_df = eval_df.append({
        'Loss': float(result.group(1)),
        'Acc': float(result.group(2))
    }, ignore_index=True)

eval_df['Steps'] = [i*5000 for i in range(1, len(eval_df)+1)]

print(train_df.head())
print(eval_df.head())

from matplotlib.ticker import FuncFormatter

train_df_subset = train_df[train_df['Steps'] % 5000 == 0]

plt.figure(figsize=(12, 8))

sns.lineplot(data=train_df_subset, x='Steps', y='Acc', label='Train accuracy')

sns.lineplot(data=eval_df, x='Steps', y='Acc', color='r', label='Validation accuracy')

plt.xlabel('Steps', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2)

max_acc_index_train = train_df_subset['Acc'].idxmax()
max_acc_train = train_df_subset.loc[max_acc_index_train, 'Acc']
max_step_train = train_df_subset.loc[max_acc_index_train, 'Steps']
plt.scatter(max_step_train, max_acc_train, color='orange')
plt.annotate(f'{max_acc_train:.4f}', (max_step_train, max_acc_train), textcoords="offset points", xytext=(-10,10), ha='center', bbox=bbox_props)


max_acc_index_eval = eval_df['Acc'].idxmax()
max_acc_eval = eval_df.loc[max_acc_index_eval, 'Acc']
max_step_eval = eval_df.loc[max_acc_index_eval, 'Steps']
plt.scatter(max_step_eval, max_acc_eval, color='orange')
plt.annotate(f'{max_acc_eval:.4f}', (max_step_eval, max_acc_eval), textcoords="offset points", xytext=(-10,10), ha='center', bbox=bbox_props)

plt.title('TP-Transformer \n n_heads=8  n_layers=6', fontsize=14, loc='center')

plt.legend()

plt.show()


min_loss_index_train = train_df_subset['Loss'].idxmin()
min_loss_train = train_df_subset.loc[min_loss_index_train, 'Loss']
min_loss_index_eval = eval_df['Loss'].idxmin()
min_loss_eval = eval_df.loc[min_loss_index_eval, 'Loss']

print(min_loss_train, min_loss_eval)

