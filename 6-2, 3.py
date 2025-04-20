from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#데이터 불러오기
iris = load_iris()
X = iris.data
y = iris.target

#PCA 수행
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#PCA 결과 시각화
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y,cmap='viridis',s=5,edgecolor='k')
plt.xlabel('주성분 1 (가장 중요)')
plt.ylabel('주성분 2 (두 번쨰로 중요)')
plt.title('Iris 데이터 PCA')
plt.grid()
plt.show()