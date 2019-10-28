import preprocessing
import pandas as pd
import models.Machine_Learning
from models.Machine_Learning.TF_IDF import TF_IDF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("Data loading...")
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/development.csv')
print("Data loading is done!")

print("Sentence cut...")
train_sentence = preprocessing.data_preprocessing(train_df['joke'])
test_sentence = preprocessing.data_preprocessing(test_df['joke'])
print("Sentence cut is done!")

X = TF_IDF(train_sentence)
y = train_df['label']
test = TF_IDF(test_sentence)

"""数据切分为训练集、验证集"""
X_train, X_dev, y_train, y_dev = model_selection.train_test_split(X, y, test_size=0.2)
print("划分训练集测试集完成！")

# """Logistic Regression"""
# clf = LogisticRegression(multi_class='ovr', solver='sag', class_weight='balanced')
# clf.fit(X_train, y_train)
#
# """Decision Tree"""
# clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# clf.fit(X_train, y_train)
#
# """Random Forest"""
# clf = RandomForestClassifier(n_estimators=15)
# clf.fit(X_train, y_train)

"""GBDT"""
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# """XGBoost"""
# clf = XGBClassifier()
# clf.fit(X_train, y_train)
#
# """Lightgbm"""
# clf = LGBMClassifier()
# clf.fit(X_train, y_train)
#
# """SVM"""
# clf = SVR()
# clf.fit(X_train, y_train)
#
# """Kfold交叉验证"""
# skfold = StratifiedKFold(n_splits=5, random_state=1)
# for train_index, test_index in skfold.split(X, y):
#     clone_clf = clone(clf)
#     X_train_folds = X[train_index]
#     y_train_folds = y[train_index]
#     X_test_fold = X[test_index]
#     y_test_fold = y[test_index]
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     f1 = f1_score(y_test_fold, y_pred)
#     print("F1预测值为"+str(f1))

# """网格搜索调参，每个参数进行调整"""
# """GBDT调参作为例子"""
# param_n_estimators = {'n_estimators': range(20, 81, 10)}
# grid_search_1 = GridSearchCV(estimator=GradientBoostingClassifier(),
#                              param_grid=param_n_estimators,
#                              iid=False,
#                              scoring='roc_auc',
#                              cv=5)
# grid_search_1.fit(X_train, y_train)
# print(grid_search_1.best_params_, grid_search_1.best_score_)







