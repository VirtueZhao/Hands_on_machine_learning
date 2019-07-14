from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

moons = make_moons()
X = moons[0]
y = moons[1]

# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
# svm_clf = SVC()
#
# voting_clf = VotingClassifier(
#     estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
#     voting='hard'
# )
# voting_clf.fit(X,y)
#
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X,y)
#     y_pred = clf.predict(X)
#     print(clf.__class__.__name__, accuracy_score(y,y_pred))

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), n_estimators=500,
#     max_samples=100, bootstrap=True, n_jobs=-1
# )
# bag_clf.fit(X,y)
# y_pred = bag_clf.predict(X)

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), n_estimators=500,
#     bootstrap=True, n_jobs=-1, oob_score=True
# )
# bag_clf.fit(X,y)
# print(bag_clf.oob_score_)
# y_pred = bag_clf.predict(X)
# print(accuracy_score(y,y_pred))

# rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
# rnd_clf.fit(X,y)
#
# y_pred_rf = rnd_clf.predict(X)

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
#     n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1
# )

# iris = load_iris()
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rnd_clf.fit(iris["data"], iris["target"])
# for name,score in zip(iris["feature_names"], rnd_clf.feature_importances_):
#     print(name, score)