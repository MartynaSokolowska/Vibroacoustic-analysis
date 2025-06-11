from feature_extraction import get_all_features
from classification import reduce_dimensionality_UMAP, fit_classifier, evaluate_classifier
from vae_reduction import reduce_dimensionality_VAE
from visualisation import plot2D, show_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    features_raw, labels = get_all_features("results_denoised")
    scaler = StandardScaler()
    features = scaler.fit_transform(features_raw)

    #X_2d = reduce_dimensionality_UMAP(features, 15)
    X_2d = reduce_dimensionality_VAE(features, latent_dim=10, epochs=50)
    # plot2D(X_2d, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X_2d, labels, test_size=0.3, random_state=42)
    clf = fit_classifier(X_train, y_train)
    y_pred = evaluate_classifier(clf, X_test, y_test)
    show_confusion_matrix(y_pred, y_test, display_labels=clf.classes_)
