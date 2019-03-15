# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:45:14 2019

@author: chari
"""

#%%
# import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


#%%
# function that generates the sum of squared distances of samples to their closest center in kmeans clustering; aids in finding the best number of k
def kmeans_parameter_search(df, kclusters):
    sse = {}
    for k in kclusters:
        kmeans = KMeans(n_clusters = k, max_iter = 1000, n_init = 25).fit(df)
        sse[k] = kmeans.inertia_
    
    return sse


#%%
# function to plot SSEs and use "elbow criterion" to determine best number of k
# look for where the SSE decreases abruptly
def plot_kmeans_sse(sse):
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


