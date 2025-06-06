from feature_extraction import get_all_features
from classification import reduce_dimensionality_UMAP, fit_classifier, evaluate_classifier
from visualisation import plot_umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    features_raw, labels = get_all_features("results_denoised")
    scaler = StandardScaler()
    features = scaler.fit_transform(features_raw)

    X_2d = reduce_dimensionality_UMAP(features)
    plot_umap(X_2d, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X_2d, labels, test_size=0.3, random_state=42)
    clf = fit_classifier(X_train, y_train)
    evaluate_classifier(clf, X_test, y_test)
