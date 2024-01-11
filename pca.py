import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from PIL import Image


image_dim = (400, 400)
image_path = "FVC2002_dataset/101/101_1.tif"

image = Image.open(image_path)
image = image.resize(image_dim, Image.LANCZOS)
print(image.size)
print(image)
plt.imshow(image, cmap="gray")
plt.show(block=False)

# Convert the image to a NumPy array
image_array = np.array(image)

pca = PCA()
pca.fit(image)

# Getting the cumulative variance
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu > 95)
print("Number of components explaining 95% variance: " + str(k))

plt.figure(figsize=[10, 5])
plt.title("Cumulative Explained Variance explained by the components")
plt.ylabel("Cumulative Explained variance")
plt.xlabel("Principal components")
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=95, color="r", linestyle="--")
ax = plt.plot(var_cumu)
plt.show(block=False)


# Function to reconstruct and plot image for a given number of components
def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_array))
    image_recon = image_recon.astype(image_array.dtype)
    image_recon = Image.fromarray(image_recon)
    plt.imshow(image_recon, cmap=plt.cm.gray)
    plt.title("Reconstructed image using {} PCA components".format(k))
    plt.show(block=False)

    if k == 100:
        print(image_recon)


ks = [40, 60, 80, 100, 120, 140]

plt.figure(figsize=[15, 9])

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plot_at_k(ks[i])
    plt.title("Components: " + str(ks[i]))

plt.show()
