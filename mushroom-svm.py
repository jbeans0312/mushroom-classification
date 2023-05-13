import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def plot_importance(classifier, feats):
    imp = classifier.coef_[0]
    imp, feats = zip(*sorted(zip(imp, feats)))
    plt.barh([x for x in range(len(imp))], imp, align='edge')
    plt.yticks(range(len(feats)), feats)
    plt.title('Feature Importance (LINEAR)')
    #plt.show()


# separate data into X and Y
data = pd.read_csv('mushrooms.csv', names=['class', 'cap-shape', 'cap-surface',
                                           'cap-color', 'bruises', 'odor', 'gill-attachment',
                                           'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                                           'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                           'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                           'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                                           'population', 'habitat'])
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# preprocess the dataframe
ppX = X.apply(lambda x: pd.factorize(x)[0])
ppX = ppX.astype('int')

ppY = Y.replace({'p': 1, 'e': 0})
ppY = ppY.astype('int')

# separate data into training and test
X_train, X_test, y_train, y_test = \
    train_test_split(ppX, ppY, test_size=0.3, random_state=0)

# create and fit our models
clf_lin = svm.SVC(kernel='linear', C=1.0)
clf_lin.fit(X_train, y_train)

print("===============LINEAR ANALYSIS===============")
# cross validation
# set cv equal to k as in K-fold
scores = cross_val_score(clf_lin, X_train, y_train, scoring='accuracy', cv=10)
print("\n==Cross Validation==")
print("CV yields %0.4f accuracy with a std dev of %0.4f" % (scores.mean(), scores.std()))

# generate prediction
print("\n==Testing Prediction==")
print("Testing yields %0.4f accuracy" % clf_lin.score(X_test, y_test))
prediction = clf_lin.predict(X_test)

# find the most informative features
features = ['cap-shape', 'cap-surface',
            'cap-color', 'bruises', 'odor', 'gill-attachment',
            'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
            'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
            'population', 'habitat']
print("\n==Feature Importance==")
for i, v in enumerate(clf_lin.coef_[0]):
    print('Feature: %s, Score: %.5f' % (features[i], v))

# plot the importance of features
print("\nPlotting Feature Importance...")
plot_importance(clf_lin, features)

# recreate test dataframe and save dataframes to txt file for plot.py
np.savetxt('./test_data.txt', X_test.values, fmt='%d')
np.savetxt('./svm_prediction.txt', prediction, fmt='%d')
