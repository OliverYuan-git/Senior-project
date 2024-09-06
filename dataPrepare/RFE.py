import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import matplotlib.pyplot as plt

red_path = 'winequality-red-corrected.csv'
red_data = pd.read_csv(red_path)
X_red = red_data.drop(columns=['quality'])
y_red = red_data['quality']

white_path = 'winequality-white-corrected.csv'  
white_data = pd.read_csv(white_path)
X_white = white_data.drop(columns=['quality'])
y_white = white_data['quality']

# Use RFE with SVM
model = SVC(kernel="linear")
rfe_red = RFE(estimator=model, n_features_to_select=5)
fit_red = rfe_red.fit(X_red, y_red)
ranking_red = pd.DataFrame({'Feature': X_red.columns, 'Ranking': fit_red.ranking_}).sort_values(by='Ranking')

rfe_white = RFE(estimator=model, n_features_to_select=5)
fit_white = rfe_white.fit(X_white, y_white)
ranking_white = pd.DataFrame({'Feature': X_white.columns, 'Ranking': fit_white.ranking_}).sort_values(by='Ranking')

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].bar(ranking_red['Feature'], ranking_red['Ranking'])
axes[0].set_title('Red Wine Feature Ranking with SVM')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Ranking')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(ranking_white['Feature'], ranking_white['Ranking'])
axes[1].set_title('White Wine Feature Ranking with SVM')
axes[1].set_xlabel('Features')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
