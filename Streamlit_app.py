import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
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
def perform_dynamic_clustering(df_scaled, algorithm, k=None, eps=None, min_samples=None, damping=None, preference=None, n_components=None, bandwidth=None, bin_seeding=None, cluster_all=None):
    pca = PCA(n_components=n_components)
    df_pca_dynamic  = pca.fit_transform(df_scaled)
    
    if algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Mean Shift":
        model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding, cluster_all=cluster_all)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Gaussian Mixture":
        model = GaussianMixture(n_components=k, random_state=42)
        model.fit(df_pca_dynamic)
        labels = model.predict(df_pca_dynamic)
    elif algorithm == "Agglomerative Clustering":
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "OPTICS":
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Affinity Propagation":
        model = AffinityPropagation(damping=damping, preference=preference)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "BIRCH":
        model = Birch(n_clusters=k)
        labels = model.fit_predict(df_pca_dynamic)
    elif algorithm == "Spectral Clustering":
        model = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
        labels = model.fit_predict(df_pca_dynamic)
    else:
        return None, None, None, None

    # Calculate Silhouette Score and Davies-Bouldin Index
    if len(set(labels)) > 1:
        silhouette = silhouette_score(df_pca_dynamic, labels)
        db_index = davies_bouldin_score(df_pca_dynamic, labels)
        calinski_score = calinski_harabasz_score(df_pca_dynamic, labels)
        dunn_index_score = dunn_index(df_pca_dynamic, labels)
    else:
        silhouette, db_index, calinski_score, dunn_index_score = -1, -1, -1, -1
    
    return df_pca_dynamic, labels, silhouette, db_index

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
        return None, None, None, None

    # Calculate Silhouette Score and Davies-Bouldin Index
    if len(set(labels)) > 1:
        silhouette = silhouette_score(df_pca, labels)
        db_index = davies_bouldin_score(df_pca, labels)
    else:
        silhouette, db_index = -1, -1
    
    return df_pca_dynamic, labels, silhouette, db_index, calinski_score, dunn_index_score

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
            df_pca, labels, silhouette, db_index = perform_static_clustering(df_scaled, algorithm)
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            plot_clusters(df_pca, labels, f"{algorithm} Clustering")
    
    if method == "Custom Clustering":
        n_components = st.slider("Select Number of PCA Components", 2, 5, 2)
        algorithm = st.selectbox("Select Clustering Algorithm", ["DBSCAN", "Mean Shift", "Gaussian Mixture", "Agglomerative Clustering", "OPTICS", "HDBSCAN", "Affinity Propagation", "BIRCH", "Spectral Clustering"])
        
        # Set up parameter sliders based on the algorithm
        if algorithm in ["Gaussian Mixture", "Agglomerative Clustering", "BIRCH", "Spectral Clustering"]:
            k = st.slider("Select Number of Clusters", 2, 10, 4)
        else:
            k = None
        eps = st.slider("Select Epsilon (eps) Value", 0.1, 5.0, 0.5, step=0.1) if algorithm == "DBSCAN" else None
        min_samples = st.slider("Select Min Samples", 1, 20, 10) if algorithm in ["DBSCAN", "OPTICS", "HDBSCAN"] else None
        damping = st.slider("Select Damping Value", 0.5, 1.0, 0.9) if algorithm == "Affinity Propagation" else None
        preference = st.slider("Select Preference Value", -100, -50, -50) if algorithm == "Affinity Propagation" else None

        if algorithm == "Mean Shift":
            bandwidth = st.selectbox("Select Bandwidth", np.linspace(0.1, 1.5, 20))
            bin_seeding = st.selectbox("Bin Seeding", [True, False])
            cluster_all = st.selectbox("Cluster All", [True, False])
        else:
            bandwidth = None
            bin_seeding = None
            cluster_all = None
        
        if st.button("Run Clustering (Custom)"):
             df_pca_dynamic, labels, silhouette, db_index, calinski_score, dunn_index_score = perform_dynamic_clustering(df_scaled, algorithm, k, eps, min_samples, damping, preference, n_components, bandwidth, bin_seeding, cluster_all)
            st.write(f"### {algorithm} Clustering Results")
            st.write(f"Silhouette Score: {silhouette:.6f}")
            st.write(f"Davies-Bouldin Index: {db_index:.6f}")
            st.write(f"Calinski-Harabasz Score: {calinski_score:.2f}")
            st.write(f"Dunn Index: {dunn_index_score:.4f}")
            plot_clusters(df_pca_dynamic, labels, f"{algorithm} Clustering")

if __name__ == "__main__":
    main()
