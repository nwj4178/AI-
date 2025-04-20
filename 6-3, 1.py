from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data[:2000], mnist.target[:2000]

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(10,8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='tab10', s=10)
plt.title("MNIST 데이터 t-SNE 시각화 (2000개 샘플)")
plt.colorbar()
plt.show()