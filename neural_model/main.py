import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessario per il 3D
from auto_encoder import AutoEncoder
from sklearn.decomposition import PCA




path = "..\data\dataset\\normalized_dataset2.npz"  # cambia con il tuo file
d = np.load(path, allow_pickle=True)

data = d["data"]
mask = d["mask"]

# print("Data shape:", data.shape)
# print("Mask shape:", mask.shape)
# print("First data sample (first 50 elements):")
# print(data[0][:50])
# print(mask[0][:50])



model = AutoEncoder(data.shape[1], latent_dim=64)
# model.train(
#         optimizer=torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.0001),
#         epochs=10,
#         input=data,
#         mask=mask
#     )   

# torch.save(model.encoder.state_dict(), "Pesi/encoder5.pth")

model.encoder.load_state_dict(torch.load(".\\Pesi\\encoder4.pth", map_location="cpu"))


avg = []
with torch.no_grad():

    for elem in data[:1000]:
        z = model.forward(torch.tensor(np.atleast_1d(elem), dtype=torch.float32),True)
        avg.append(z.cpu().numpy())

Z = np.array(avg)  # shape: (N, 32)



# Riduzione a 3 dimensioni
Z3 = PCA(n_components=3).fit_transform(Z)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()

# Riduzione a 2 dimensioni
Z2 = PCA(n_components=2).fit_transform(Z)

# Plot 2D
plt.figure()
plt.scatter(Z2[:, 0], Z2[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Spazio latente (PCA 2D)")
plt.grid(True)
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(Z)

# Riduciamo a 2 dimensioni
pca = PCA(n_components=2)
Z_2d = pca.fit_transform(Z)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cluster nello spazio latente (PCA)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# Clustering nello spazio latente
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(Z)

# Riduzione a 3 dimensioni
pca = PCA(n_components=3)
Z_3d = pca.fit_transform(Z)

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    Z_3d[:, 0],
    Z_3d[:, 1],
    Z_3d[:, 2],
    c=clusters
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Cluster nello spazio latente (PCA 3D)")

plt.colorbar(sc, label="Cluster")
plt.show()