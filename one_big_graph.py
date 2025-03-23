import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

models = ['Transformer 8h-6l', 'TP-Transformer 4h-3l', 'TP-Transformer 6h-6l', 'TP-Transformer 8h-6l', 'TP-Transformer 3h-8l', 'TP-Transformer 4h-8l', 'TP-Transformer 8h-8l']

train_accuracy = [0.8567, 0.7638, 0.8303, 0.8616, 0.8698, 0.8686, 0.8653]
val_accuracy = [0.7767, 0.6692, 0.7490, 0.7822, 0.7925, 0.7918, 0.7892]

data = {'Model': models, 'Training Accuracy': train_accuracy, 'Validation Accuracy': val_accuracy}
df = pd.DataFrame(data)

sns.set_style()

sns.scatterplot(data=df, x='Model', y='Training Accuracy', color='blue', marker='o', label='Training Accuracy')
sns.scatterplot(data=df, x='Model', y='Validation Accuracy', color='orange', marker='s', label='Validation Accuracy')

for i in range(len(models)):
    plt.plot([models[i], models[i]], [train_accuracy[i], val_accuracy[i]], 'k--')

plt.xticks(rotation=45)

plt.ylabel('Accuracy')

plt.legend()

plt.show()