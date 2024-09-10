import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

red_wine_path = 'winequality-red-corrected.csv'
white_wine_path = 'winequality-white-corrected.csv'
red_wine_data = pd.read_csv(red_wine_path)
white_wine_data = pd.read_csv(white_wine_path)

red_wine_data['target'] = (red_wine_data['quality'] >= 7).astype(int)
white_wine_data['target'] = (white_wine_data['quality'] >= 7).astype(int)

red_wine_data['type'] = 'Red Wine'
white_wine_data['type'] = 'White Wine'
combined_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

plt.figure(figsize=(14, 10))
for i, col in enumerate(red_wine_data.columns[:-3]):
    plt.subplot(4, 3, i + 1)
    sns.histplot(data=combined_data, x=col, hue='type', kde=True, element='step', stat='density', common_norm=False)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=combined_data, x='target', hue='type')
plt.title('Target Variable Distribution')
plt.xlabel('Target (0: Low Quality, 1: High Quality)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 10))
for i, col in enumerate(red_wine_data.columns[:-3]):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(data=combined_data, x='type', y=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
for i, col in enumerate(red_wine_data.columns[:-3]):
    plt.subplot(4, 3, i + 1)
    sns.violinplot(data=combined_data, x='type', y=col, split=True, inner="quart")
    plt.title(f'Violin Plot of {col}')
plt.tight_layout()
plt.show()


