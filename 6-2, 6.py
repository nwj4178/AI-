from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

faces = fetch_olivetti_faces()
X = faces.data
images = faces.images

ica = FastICA(n_components=25)
ica.fit(X)
components_ica = ica.components_

pca = PCA(n_components=25)
pca.fit(X)
components_pca = pca.components_

def plot_components(components, title):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(components[i].reshape(64, 64), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plot_components(components_ica, "ICA Components (25개)")
plot_components(components_pca, "PCA Components (25개)")