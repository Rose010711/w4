The code below performs K-means clustering on satellite imagery data from the Sentinel-2 mission, specifically targeting three different spectral bands (B4, B3, B2) that represent red, green, and blue wavelengths, respectively. Here's a breakdown of the logic step by step:

Path Specification: The base path where the satellite images are stored is defined. This path is to a specific directory containing images from a Sentinel-2 satellite pass.

Band Paths Dictionary: A dictionary named bands_paths is created to map each spectral band (B4, B3, B2) to its corresponding file location. These file paths are constructed by concatenating the base path with the specific filename for each band.

Reading and Stacking Band Images:

An empty list named band_data is initialized to store the data from each band.
A for-loop iterates over the specified bands (B4, B3, B2) in sequence.
Within the loop, each band's image file is opened using rasterio.open, and the first band of data (.read(1)) is read and appended to the band_data list.
Creating a Valid Data Mask:

The individual bands are stacked together into a 3-dimensional array (band_stack) using numpy.dstack, aligning them along a new third axis.
A valid data mask (valid_data_mask) is created by checking where all bands have non-zero values (np.all(band_stack > 0, axis=2)), which helps to exclude pixels with invalid or missing data across any of the bands.
Reshaping for K-means Clustering:

The valid data pixels are extracted and reshaped into a 2D array X, where each row corresponds to a pixel and each column to a band's intensity value. This is done by masking the band_stack with valid_data_mask and then reshaping it.
K-means Clustering:

The KMeans clustering algorithm from sklearn.cluster is applied to the 2D array X, specifying 2 clusters (n_clusters=2) and a fixed random state for reproducibility.
The algorithm assigns each pixel to one of the two clusters, resulting in a set of labels (labels).
Creating and Populating the Labels Image:

An empty array (labels_image) is created, initially filled with a no-data value (e.g., -1). This array has the same spatial dimensions as the original band images but is intended to store cluster labels.
The cluster labels are placed into labels_image at the positions corresponding to valid data pixels, as indicated by the valid_data_mask.
Plotting the Result:

The final step involves visualizing the clustering result using matplotlib.pyplot. The labels_image array is displayed as an image where different clusters are represented by different colors in the 'viridis' colormap.
A color bar is added to indicate the cluster labels, and a title is set for the plot.
In summary, this script demonstrates how to perform and visualize K-means clustering on multispectral satellite imagery to segregate the landscape into two distinct clusters based on the spectral characteristics of the pixels in the red, green, and blue bands.


import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/AI4EO/Week_4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

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


The code below is used to analyze and compare the distribution of intensity values across the blue, green, and red spectral bands of a satellite image, specifically focusing on a subset of the data and within a defined intensity range. This can help in understanding the variability and characteristics of different areas within the satellite image based on their spectral properties.

plt.hist(band_stack[::100, ::100, 2].ravel(), bins=100, range=(4000,10000),label='green')  #
plt.hist(band_stack[::100, ::100, 1].ravel(), bins=100,range=(4000,10000), label='red')    #
plt.hist(band_stack[::100, ::100, 0].ravel(), bins=100, range=(4000,10000),label='blue')   #
plt.legend()


This code is a demonstration of applying machine learning (GMM clustering) to satellite imagery for the purpose of segmenting the image into distinct areas based on the spectral characteristics of the pixels.

import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/AI4EO/Week_4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()


Before delving into the modeling process, it's crucial to preprocess the data to ensure compatibility with our analytical models. This involves transforming the raw data into meaningful variables, such as peakniness and stack standard deviation (SSD), etc.


! pip install netCDF4

!pip install Basemap

!pip install cartopy








Next code is designed to process and analyze satellite radar altimeter data, focusing on waveform analysis to extract features such as peakiness and spread. It consists of several functions each serving a specific purpose:

1，peakiness: Calculates the "peakiness" of waveforms, a measure of how peaked or sharp a waveform is. This function also allows for visualization of individual waveforms and their peakiness calculation.

2，unpack_gpod: Handles the extraction and interpolation of variables from satellite data. It's designed to work with data that have different temporal resolutions, expanding them to a uniform 20Hz frequency for consistent analysis.

3，calculate_SSD: Computes the Sum of Squared Differences (SSD) for waveforms, which is a statistical measure used to analyze the distribution of power within a waveform. It fits a Gaussian model to the waveforms to estimate this parameter, indicating the spread of the return signal.

The script utilizes various scientific computing libraries such as NumPy for array manipulations, Matplotlib for plotting, SciPy for statistical functions, and sklearn for clustering and data scaling. It starts by defining functions that can be used to analyze waveform data for physical properties, interpolating data to match sampling frequencies, and fitting statistical models to understand waveform characteristics. These functions are intended to be used on data from satellite radar altimeters, which are instruments used to measure the distance from the satellite to the Earth's surface with high precision. This analysis can help in understanding surface characteristics, such as sea ice, ocean waves, and land topography.






#
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree

import pyproj
# import cartopy.crs as ccrs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

#=========================================================================================================
#===================================  SUBFUNCTIONS  ======================================================
#=========================================================================================================

#*args and **kwargs allow you to pass an unspecified number of arguments to a function,
#so when writing the function definition, you do not need to know how many arguments will be passed to your function
#**kwargs allows you to pass keyworded variable length of arguments to a function.
#You should use **kwargs if you want to handle named arguments in a function.
#double star allows us to pass through keyword arguments (and any number of them).
def peakiness(waves, **kwargs):

    "finds peakiness of waveforms."

    #print("Beginning peakiness")
    # Kwargs are:
    #          wf_plots. specify a number n: wf_plots=n, to show the first n waveform plots. \

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    print("Running peakiness function...")

    size=np.shape(waves)[0] #.shape property is a tuple of length .ndim containing the length of each dimensions
                            #Tuple of array dimensions.

    waves1=np.copy(waves)

    if waves1.ndim == 1: #number of array dimensions
        print('only one waveform in file')
        waves2=waves1.reshape(1,np.size(waves1)) #numpy.reshape(a, newshape, order='C'), a=array to be reshaped
        waves1=waves2

    # *args is used to send a non-keyworded variable length argument list to the function
    def by_row(waves, *args):
        "calculate peakiness for each waveform"
        maximum=np.nanmax(waves)
        if maximum > 0:

            maximum_bin=np.where(waves==maximum)
            #print(maximum_bin)
            maximum_bin=maximum_bin[0][0]
            waves_128=waves[maximum_bin-50:maximum_bin+78]

            waves=waves_128

            noise_floor=np.nanmean(waves[10:20])
            where_above_nf=np.where(waves > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum=np.nanmax(waves[where_above_nf])
                total=np.sum(waves[where_above_nf])
                mean=np.nanmean(waves[where_above_nf])
                peaky=maximum/mean

            else:
                peaky = np.nan
                maximum = np.nan
                total = np.nan

        else:
            peaky = np.nan
            maximum = np.nan
            total = np.nan

        if 'maxs' in args:
            return maximum
        if 'totals' in args:
            return total
        if 'peaky' in args:
            return peaky

    peaky=np.apply_along_axis(by_row, 1, waves1, 'peaky') #numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    if 'wf_plots' in kwargs:
        maximums=np.apply_along_axis(by_row, 1, waves1, 'maxs')
        totals=np.apply_along_axis(by_row, 1, waves1, 'totals')

        for i in range(0,kwargs['wf_plots']):
            if i == 0:
                print("Plotting first "+str(kwargs['wf_plots'])+" waveforms")

            plt.plot(waves1[i,:])#, a, col[i],label=label[i])
            plt.axhline(maximums[i], color='green')
            plt.axvline(10, color='r')
            plt.axvline(19, color='r')
            plt.xlabel('Bin (of 256)')
            plt.ylabel('Power')
            plt.text(5,maximums[i],"maximum="+str(maximums[i]))
            plt.text(5,maximums[i]-2500,"total="+str(totals[i]))
            plt.text(5,maximums[i]-5000,"peakiness="+str(peaky[i]))
            plt.title('waveform '+str(i)+' of '+str(size)+'\n. Noise floor average taken between red lines.')
            plt.show()


    return peaky

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================


def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD
