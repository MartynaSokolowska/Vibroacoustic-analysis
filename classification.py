import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, silhouette_score, davies_bouldin_score


def reduce_dimensionality_UMAP(X, components):
    reducer = umap.UMAP(n_neighbors=6, min_dist=0.1, spread=0.6, n_components=components, random_state=42)
    return reducer.fit_transform(X)


def fit_classifier(X, y, k=3):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    return preds


def calculate_silhouette_score(X, y):
    return silhouette_score(X, y)


def calculate_davies_bouldin_score(X, y):
    return davies_bouldin_score(X, y)


def print_scores(X, y):
    silhouette = calculate_silhouette_score(X, y)
    db_score = calculate_davies_bouldin_score(X, y)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")
