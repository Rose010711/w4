# K-means clustering algorithm

The K-means clustering on satellite imagery by data code from the Sentinel-2 mission, specifically targeting three different spectral bands.
K-means clustering is an intuitive algorithm that uncovers and predicts inherent groupings within data without assuming any prior informational data about the data's distribution (Sinaga & Yang, 2020). K-means clustering was chosen because of its useful ability to deal with unknown data structures, especially when searching data. The algorithm is simple and efficient to scale to data sets of all sizes, so it's easy to use it in many practical applications.  

# Unsupervised Learning: Gaussian Mixture Models (GMM)

At first, Gaussian Mixture Models (GMMs) are probabilistic tools that identify subpopulations within a dataset, assuming these groups follow a normal distribution (Reynolds, 2009). 
In Jupyter Book, it shows that GMMs apply soft clustering by assigning probabilities of belonging to each cluster to individual data points, capturing the inherent uncertainties of the data—making them especially useful when the data’s structure is unknown. A key advantage of GMMs is their versatility; they adapt to clusters of various sizes and shapes, determined by the covariance structure like spherical, diagonal, or full covariance. Moreover, GMMs utilize the Expectation-Maximization (EM) algorithm, iteratively refining the model fit to the data until convergence.

<img width="437" alt="image" src="https://github.com/Rose010711/w4/assets/161240176/e6a71e79-cf6e-4d18-971c-2567be13224a">
a very basic one

# Sentinel-2

Sentinel-2 is designed for high-resolution optical imaging. It's equipped to capture data in 13 spectral bands, with applications including monitoring land changes, agriculture, and forestry. Sentinel-2's ability to provide frequent updates due to its high revisit rate makes it valuable for tracking the gradual changes on the Earth's surface (Cazzaniga et al., 2019)​​.

The code below is used to analyze and compare the distribution of intensity values across the blue, green, and red spectral bands of a satellite image, specifically focusing on a subset of the data and within a defined intensity range. This can help in understanding the variability and characteristics of different areas within the satellite image based on their spectral properties.

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

<img width="447" alt="image" src="https://github.com/Rose010711/w4/assets/161240176/171fa3cc-8a08-465b-a124-456849393daa">

# Sentinel-3
Sentinel-3, on the other hand, focuses more broadly on Earth’s oceans, land, ice, and atmosphere, supporting long-term global monitoring. It's instrumental in observing sea surface level, land surface temperature, vegetation, and ocean currents among other parameters (Cazzaniga et al., 2019)​​.
<img width="471" alt="image" src="https://github.com/Rose010711/w4/assets/161240176/a19f8be2-9c9e-499c-943d-1678b4a6b9d4">

This graph shows the results of a data cluster using a Gaussian mixture model (GMM), where each color represents a function that may correspond to a different data feature or category. Here, clusters_gmm == 0 might represent a collection of data points identified as a specific category, such as "sea ice". Each peak in the graph reflects the distribution of different data points within that cluster, with the most prominent peaks representing the most common or typical characteristic responses in that category. If the data comes from the Sentinel-3 satellite, these functions may represent spectral properties of different areas of sea ice or measures of parameters such as sea ice thickness or temperature. Through such analysis, we are able to more accurately understand and monitor changes in sea ice, which is of great value for climate research and ocean navigation.

Sentinel-3 captures the spectral properties of sea ice, and the GMM can process that data to distinguish between clusters or features - for example, between dense areas of sea ice and open waterways what represents it for the leads.

# Sentinel-2 & 3, 
they are both in the part of European Space Agency's Copernicus program, and utlize the Gaussian Mixture Models for various remote sensing applications by providing the raw data.

# what's more, for installation
! pip install netCDF4

!pip install Basemap

!pip install cartopy

# Reference
Sinaga, K. P., & Yang, M. S. (2020). Unsupervised K-means clustering algorithm. IEEE access, 8, 80716-80727.
Reynolds, D. A. (2009). Gaussian mixture models. Encyclopedia of biometrics, 741(659-663).
Cazzaniga, I., Bresciani, M., Colombo, R., Della Bella, V., Padula, R., & Giardino, C. (2019). A comparison of Sentinel-3-OLCI and Sentinel-2-MSI-derived Chlorophyll-a maps for two large Italian lakes. Remote sensing letters, 10(10), 978-987.
Geol0069 AI4EO Lecture
