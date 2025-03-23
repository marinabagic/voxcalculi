import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Model': [], 'Module': [], 'Accuracy': []}
with open('C:\\Users\\anteg\\Desktop\\eval_extrapolation.txt', 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 16):
        model = lines[i].strip() 
        for j in range(i+1, i+16):
            module, accuracy = lines[j].strip().split(' accuracy: ')
            data['Model'].append(model)
            data['Module'].append(module)
            data['Accuracy'].append(float(accuracy))

df = pd.DataFrame(data)

plt.figure(figsize=(10, 10)) 
heatmap_data = df.pivot_table(index='Module', columns='Model', values='Accuracy')
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
plt.xlabel('Model')
plt.ylabel('Module')
plt.xticks(rotation=45)
plt.title('Model Performance')
plt.tight_layout()
plt.show()