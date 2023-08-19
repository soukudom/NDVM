from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from weles.classifiers import fkNN

X, y = load_iris(return_X_y=True)

clf1 = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
clf2 = fkNN(k=5, p=0.5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1410)

clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)
print(accuracy_score(y_test, pred1))

clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)
print(accuracy_score(y_test, pred2))
