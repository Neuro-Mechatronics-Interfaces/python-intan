import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca(X, n_components=15, variance_threshold=0.95, show_plot=False, verbose=False):
    """ Implements PCA on a dataset with the number of components specified.
    Args:
        X: Input data of shape (samples, channels)
        n_components: Number of components to keep."""

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # Shape: (samples, components)
    X_reconstructed = pca.inverse_transform(X_pca).T  # Shape: (channels, samples)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find how many components reach 95% variance
    n_pca_95 = np.argmax(cumulative_variance >= variance_threshold) + 1
    if verbose:
        print(f"Number of components to retain 95% variance: {n_pca_95}")

    if show_plot:
        plt.plot(cumulative_variance * 100)
        plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance Explained (%)")
        plt.title("Screen Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

    pca_k = PCA(n_components=n_pca_95)
    X_pca_k = pca_k.fit_transform(X)
    X_reconstructed_k = pca_k.inverse_transform(X_pca_k).T  # Shape: [channels, samples]

    return X_reconstructed_k, n_pca_95
