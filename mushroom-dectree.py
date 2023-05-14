import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score

# read the data
df = pd.read_csv('mushrooms.csv', names=['class', 'cap-shape', 'cap-surface',
                                           'cap-color', 'bruises', 'odor', 'gill-attachment',
                                           'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                           'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                           'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                           'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                                           'population', 'habitat'])

# split the data
y = df['class']
x = df.drop(['class'], axis=1)
x = pd.get_dummies(x)
y = pd.get_dummies(y)


# testing and training data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# creating decision tree using entropy without pruning
data_en = DecisionTreeClassifier(random_state=0)
data_en.fit(X_train, y_train)

# plot the tree
tree.plot_tree(data_en.fit(X_train, y_train))
#plt.show()

# pruning the tree
param_grid = {
 'max_depth':[2,5,8,11],
 'criterion':['gini', 'entropy'],
 'min_impurity_decrease': [0.0001, 0.0005, 0.001, 0.005, 0.01]
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5, n_jobs=1)
gridSearch.fit(X_train, y_train)
print('Best score: ', gridSearch.best_score_)
print('Best parameters: ', gridSearch.best_params_)

pruned_tree = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_decrease=0.0001)
tree.plot_tree(pruned_tree.fit(X_train, y_train))
plt.show()