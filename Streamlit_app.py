import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering, OPTICS, AffinityPropagation, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture

# Links to your .pkl files on GitHub
data_file_url = "https://github.com/Phua0414/AssignmentMachineLearning/releases/download/Tag-1/data.pkl"
models_file_url = "https://github.com/Phua0414/AssignmentMachineLearning/releases/download/Tag-1/all_models.pkl"

# File paths
data_file_path = "data.pkl"
models_file_path = "all_models.pkl"

# Function to download file from GitHub
def download_file_from_github(url, destination):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)

# Download the files from GitHub
download_file_from_github(data_file_url, data_file_path)
download_file_from_github(models_file_url, models_file_path)

# Load data and models from pickle files
def load_data_and_models():
    with open(data_file_path, "rb") as data_file:
        data = pickle.load(data_file)
    
    with open(models_file_path, "rb") as model_file:
        models = pickle.load(model_file)
    
    return data, models

# Load the data and models
data, models = load_data_and_models()

# Extract individual models
dbscan_model = models['dbscan_model']
mean_shift_model = models['mean_shift_model']
agg_clustering_model = models['agg_clustering_model']
optics_model = models['optics_model']
aff_prop_model = models['aff_prop_model']
birch_model = models['birch_model']
spectral_model = models['spectral_model']
gmm_model = models['gmm_model']
hdbscan_model = models['hdbscan_model']
df_pca = data['df_pca']
df_scaled = data['processed_data']

# Function to compute Dunn Index
def dunn_index(X, labels):
    unique_clusters = list(set(labels))
    if len(unique_clusters) < 2:
        return -1
    cluster_centers = [X[labels == k].mean(axis=0) for k in unique_clusters]
    inter_dists = cdist(cluster_centers, cluster_centers)
    np.fill_diagonal(inter_dists, np.inf)
    min_intercluster = inter_dists.min()
    max_intracluster = max([
        cdist(X[labels == k], X[labels == k]).max()
        for k in unique_clusters if len(X[labels == k]) > 1
    ])
    return min_intercluster / max_intracluster if max_intracluster != 0 else -1

# Function to perform dynamic clustering (with user-defined parameters)
def perform_dynamic_clustering(df_scaled, algorithm, num_clusters=None, eps=None, damping=None, preference=None, n_components=None, bandwidth=None, bin_seeding=None, cluster_all=None, covariance_type=None, linkage = None, metric = None, xi=None, n_neighbors=None, gamma=None, affinity=None, threshold=None, branching_factor=None, min_cluster_size=None, selection_method=None):
    pca = PCA(n_components=n_components)
    df_pca_dynamic  = pca.fit_transform(df_scaled)
    
    if algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, metric=metric)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Mean Shift":
        model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding, cluster_all=cluster_all)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Gaussian Mixture":
        model = GaussianMixture(n_components=num_clusters, covariance_type=covariance_type, random_state=42)
        model.fit(df_pca_dynamic)
        labels = model.predict(df_pca_dynamic)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage, metric=metric)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "OPTICS":
        model = OPTICS(max_eps=eps, metric=metric, xi=xi)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, cluster_selection_method=selection_method)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Affinity Propagation":
        model = AffinityPropagation(damping=damping, preference=preference, affinity=metric)
        labels = model.fit_predict(df_pca_dynamic)
        # Check if the number of labels is invalid (too many or too few clusters)
        if len(set(labels)) < 2 or len(set(labels)) >= len(df_pca_dynamic):
            return df_pca_dynamic, labels, -1, -1, -1, -1 
    elif algorithm == "BIRCH":
        model = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=num_clusters)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Spectral Clustering":
        model = SpectralClustering(
            n_clusters=num_clusters, 
            affinity=affinity, 
            n_neighbors=n_neighbors if affinity == 'nearest_neighbors' else 10,  # Only apply n_neighbors if affinity is 'nearest_neighbors'
            gamma=gamma if affinity == 'rbf' else 1.0  # Only apply gamma if affinity is 'rbf'
        )
        labels = model.fit_predict(df_pca_dynamic)
    else:
        return None, None, -1, -1, -1, -1

    # Calculate Silhouette Score and Davies-Bouldin Index
    if len(set(labels)) > 1:
        silhouette = silhouette_score(df_pca_dynamic, labels)
        db_index = davies_bouldin_score(df_pca_dynamic, labels)
        calinski_score = calinski_harabasz_score(df_pca_dynamic, labels)
        dunn_index_score = dunn_index(df_pca_dynamic, labels)
    else:
        silhouette, db_index, calinski_score, dunn_index_score = -1, -1, -1, -1
    
    return df_pca_dynamic, labels, silhouette, db_index, calinski_score, dunn_index_score

# Function to perform static clustering (using pre-trained models)
def perform_static_clustering(df_scaled, algorithm):
    if algorithm == "DBSCAN":
        model = dbscan_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "Mean Shift":
        model = mean_shift_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "Gaussian Mixture":
        model = gmm_model
        labels = model.predict(df_pca)
    elif algorithm == "Agglomerative Clustering":
        model = agg_clustering_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "OPTICS":
        model = optics_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "HDBSCAN":
        model = hdbscan_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "Affinity Propagation":
        model = aff_prop_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "BIRCH":
        model = birch_model
        labels = model.fit_predict(df_pca)
    elif algorithm == "Spectral Clustering":
        model = spectral_model
        labels = model.fit_predict(df_pca)
    else:
        return None, None, None, None, None, None

    # Calculate Silhouette Score and Davies-Bouldin Index
    if len(set(labels)) > 1:
        silhouette = silhouette_score(df_pca, labels)
        db_index = davies_bouldin_score(df_pca, labels)
        calinski_score = calinski_harabasz_score(df_pca, labels)
        dunn_index_score = dunn_index(df_pca, labels)
    else:
        silhouette, db_index = -1, -1
    
    return df_pca, labels, silhouette, db_index, calinski_score, dunn_index_score

# Function to plot clusters
def plot_clusters(df_pca, labels, title):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    st.pyplot(plt)

# Streamlit UI
def main():
    st.title("Machine Learning Clustering App")
    
    url = "https://raw.githubusercontent.com/Phua0414/AssignmentMachineLearning/main/marine-historical-2023-en.csv"
    
    # Read CSV data from GitHub
    df = pd.read_csv(url)
    st.write("### Raw Data Preview")
    st.write(df.head())
    
    # Processed data (scaled)
    st.write("### Processed Data Preview")
    st.write(df_scaled.head())

    st.write("## Clustering Method Selection")
    method = st.selectbox("Choose Clustering Method", ["Pre-trained Models", "Custom Clustering"])
    
    if method == "Pre-trained Models":
        algorithm = st.selectbox("Select Clustering Algorithm", ["DBSCAN", "Mean Shift", "Gaussian Mixture", "Agglomerative Clustering", "OPTICS", "HDBSCAN", "Affinity Propagation", "BIRCH", "Spectral Clustering"])
        
        if st.button("Run Clustering (Pre-trained Models)"):
            df_pca, labels, silhouette, db_index, calinski_score, dunn_index_score = perform_static_clustering(df_scaled, algorithm)
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            st.write(f"Calinski-Harabasz Score: {calinski_score:.2f}")
            st.write(f"Dunn Index: {dunn_index_score:.4f}")
            plot_clusters(df_pca, labels, f"{algorithm} Clustering")
    
    if method == "Custom Clustering":
        # Default values for the parameters
        bandwidth, bin_seeding, cluster_all = None, None, None
        num_clusters, k, linkage = None, None, None
        metric, eps, xi = None, None, None
        covariance_type, damping, preference = None, None, None
        n_neighbors, gamma, affinity = None, None, None
        threshold, branching_factor, min_cluster_size = None, None, None
        selection_method = None
        
        n_components = st.slider("Select Number of PCA Components", 2, 5, 2)
        algorithm = st.selectbox("Select Clustering Algorithm", ["DBSCAN", "Mean Shift", "Gaussian Mixture", "Agglomerative Clustering", "OPTICS", "HDBSCAN", "Affinity Propagation", "BIRCH", "Spectral Clustering"])


        if algorithm == "HDBSCAN":
            min_cluster_size = st.selectbox("Select Min Cluster Size", range(5, 21, 3))
            selection_method = st.selectbox("Select Selection Method", ['eom', 'leaf'])
            metric = st.selectbox("Select Metric", ['euclidean', 'manhattan'])
        
        elif algorithm == "DBSCAN":
            eps_values = np.linspace(0.02, 0.2, 10)
            eps_values_rounded = np.round(eps_values, 2)
            eps = st.selectbox("Select Epsilon (eps)", eps_values_rounded)
            metric = st.selectbox("Select Metric", ['euclidean', 'manhattan'])
        
        elif algorithm == "Spectral Clustering":
            # Spectral Clustering Parameters
            num_clusters = st.selectbox("Select Number of Clusters", range(2, 11))
            affinity = st.selectbox("Select Affinity", ['nearest_neighbors', 'rbf'])
            if affinity == 'nearest_neighbors':
                n_neighbors = st.slider("Select Number of Neighbors", 5, 20, 10)  # Range for n_neighbors
            else:
                gamma = st.slider("Select Gamma", 0.5, 2.0, 1.0) 
                
        elif algorithm == "BIRCH":
            num_clusters = st.selectbox("Select Number of Clusters", range(2, 11))
            threshold = st.selectbox("Select Threshold", [0.1, 0.2, 0.3])
            branching_factor = st.selectbox("Select Branching Factor", [25, 50])
                
        elif algorithm == "Gaussian Mixture":
            num_clusters = st.slider("Select Number of Clusters", 2, 10, 4)
            covariance_type = st.selectbox("Select Covariance Type", ['full', 'tied', 'diag', 'spherical'])

        elif algorithm == "Affinity Propagation":
            damping = st.slider("Select Damping", 0.5, 0.99, 0.5)
            preference = st.selectbox("Select Preference", [-200, -150, -100, -50])
            metric = 'euclidean'

        elif algorithm == "Agglomerative Clustering":
            num_clusters = st.slider("Select Number of Clusters", 2, 10, 4)
            linkage = st.selectbox("Select Linkage Method", ['ward', 'complete', 'average'])
            if linkage == 'ward':
                metric = 'euclidean'
            else:
                metric = st.selectbox("Select Metric", ['euclidean', 'manhattan'])

        elif algorithm == "OPTICS":
            eps = st.selectbox("Select Max Epsilon (eps)", [0.2, 2.0, 15])
            xi = st.selectbox("Select Xi", [0.02, 0.05, 0.1])
            metric = st.selectbox("Select Metric", ['euclidean', 'manhattan'])

        elif algorithm == "Mean Shift":
            bandwidth = st.slider("Select Bandwidth", 0.1, 1.5, 1.0)
            bin_seeding = st.selectbox("Bin Seeding", [True, False])
            cluster_all = st.selectbox("Cluster All", [True, False])
        
        if st.button("Run Clustering (Custom)"):
            df_pca_dynamic, labels, silhouette, db_index, calinski_score, dunn_index_score = perform_dynamic_clustering(df_scaled, algorithm, num_clusters, eps, damping, preference, n_components, bandwidth, bin_seeding, cluster_all, covariance_type, linkage, metric, xi, n_neighbors, gamma, affinity, threshold, branching_factor, min_cluster_size, selection_method)
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            st.write(f"Calinski-Harabasz Score: {calinski_score:.2f}")
            st.write(f"Dunn Index: {dunn_index_score:.4f}")
            plot_clusters(df_pca_dynamic, labels, f"{algorithm} Clustering")

if __name__ == "__main__":
    main()
