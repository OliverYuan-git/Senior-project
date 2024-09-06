import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def load_and_rank(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['quality'])
    y = data['quality']
    
    # Apply ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    ranking = pd.DataFrame({'Feature': X.columns, 'Score': scores})
    ranking = ranking.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return ranking

red_ranking = load_and_rank('winequality-red-corrected.csv')
white_ranking = load_and_rank('winequality-white-corrected.csv')


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].bar(red_ranking['Feature'], red_ranking['Score'])
axes[0].set_title('Red Wine Feature Scores (ANOVA F-test)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Score')
axes[0].tick_params(axis='x', rotation=45)
axes[1].bar(white_ranking['Feature'], white_ranking['Score'])
axes[1].set_title('White Wine Feature Scores (ANOVA F-test)')
axes[1].set_xlabel('Features')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Red Wine Feature Ranking:\n", red_ranking)
print("\nWhite Wine Feature Ranking:\n", white_ranking)
