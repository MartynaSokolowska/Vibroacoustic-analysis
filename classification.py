import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def reduce_dimensionality(X):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    return reducer.fit_transform(X)


def fit_classifier(X, y):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    return preds
