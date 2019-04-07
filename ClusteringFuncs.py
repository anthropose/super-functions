# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:45:14 2019

@author: chari
"""

#%%
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture


#%%
# function to generate AICs and BICs for gaussian mixture estimation
def gaussian_parameter_search(df, n_components, cov_type = "full"):
    AIC = {}
    BIC = {}
    if cov_type == "full":
        for n in n_components:
            gmm = GaussianMixture(n, covariance_type = "full", max_iter = 1000, 
                                  n_init = 25, 
                                  random_state = 42).fit(df)
            AIC[n] = gmm.aic(df)
            BIC[n] = gmm.bic(df)
    elif cov_type == "tied":
        for n in n_components:
            gmm = GaussianMixture(n, covariance_type = "tied", max_iter = 1000, 
                                  n_init = 25, 
                                  random_state = 42).fit(df)
            AIC[n] = gmm.aic(df)
            BIC[n] = gmm.bic(df)
    elif cov_type == "diag":
        for n in n_components:
            gmm = GaussianMixture(n, covariance_type = "diag", max_iter = 1000, 
                                  n_init = 25, 
                                  random_state = 42).fit(df)
            AIC[n] = gmm.aic(df)
            BIC[n] = gmm.bic(df)
            
    elif cov_type == "spherical":
        for n in n_components:
            gmm = GaussianMixture(n, covariance_type = "spherical", 
                                  max_iter = 1000, 
                                  n_init = 25, 
                                  random_state = 42).fit(df)
            AIC[n] = gmm.aic(df)
            BIC[n] = gmm.bic(df)
    
    return AIC, BIC


#%%
# function to plot AICs and BICs and find the lowest value to determine the best n
def plot_gaussian_AIC_BIC(AIC, BIC):
    plt.figure()
    plt.plot(list(AIC.keys()), list(AIC.values()), label = "AIC")
    plt.plot(list(BIC.keys()), list(BIC.values()), label = "BIC")
    plt.xlabel("Number of Components/Clusters")
    plt.ylabel("Information Criterion")
    plt.legend(loc = "best")


#%%
# function to generate a dataframe of the component labels (i.e., "cluster")
# and their associated posterior probabilities given the data; requires the user
# to supply a fitted gaussian model
def gaussian_probability_DF(df, gaussian_model, n_components):
    new_df = pd.DataFrame(data = gaussian_model.predict(df), 
                          index = df.index,
                          columns = ["GMM_Cluster"])
    prob_array = np.array(gaussian_model.predict_proba(df))
    for n in range(n_components):
        new_df["Prob_%d" % n] = prob_array[:, n]
    
    return new_df


#%%
# function that generates the sum of squared distances of samples to their 
# closest center in kmeans clustering; aids in finding the best number of k
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


#%%
# function to select the number of clusters using silhouette analysis and plots
def kmeans_silhouette_plot(df, kclusters):
    for k in kclusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots(figsize = (10,4))
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax.set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(df) + (k + 1) * 100])
    
        # Initialize the clusterer with k value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters = k, random_state = 10)
        cluster_labels = clusterer.fit_predict(df)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)
    
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / k)

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor = color, 
                              edgecolor = color, 
                              alpha = 0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 100  # 100 for the 0 samples
    
        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
    plt.show()
