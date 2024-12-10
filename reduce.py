import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# load data

#csv_file = 'embeddings.csv'
#post_ids = embeddings_df['id']  # Assuming 'post_id' is a column in your dataframe
#embeddings_df = pd.read_csv(csv_file, header=0, usecols=lambda column: column != 'id')
#embeddings = embeddings_df.to_numpy(dtype=np.float32)


csv_file = 'embeddings.csv'
embeddings_df = pd.read_csv(csv_file, header=0)
post_ids = embeddings_df['id']  # Assuming 'post_id' is a column in your dataframe
embeddings = embeddings_df.drop(['id'], axis=1).to_numpy(dtype=np.float32)  # Exclude the 'post_id' column from embeddings

# reduce dimensions (try 16, 32, 64, 128)
n_components = 64
pca = PCA(n_components=n_components)
reduced_embeddings = pca.fit_transform(embeddings)


# k-means on reduced embeddings
k = 20
rs=3
reduced_kmeans = KMeans(n_clusters=k, random_state=rs).fit(reduced_embeddings)
reduced_clusters = reduced_kmeans.labels_
centroids = reduced_kmeans.cluster_centers_
cluster_labels = reduced_kmeans.labels_

# Count members in each cluster
cluster_counts = np.bincount(reduced_clusters)
for i, count in enumerate(cluster_counts):
    print(f'Cluster {i}: {count} members')

# print scores

print(embeddings[0].shape[0])

silhouette_avg_reduced = silhouette_score(embeddings, reduced_clusters)
print(f'Reduced Silhouette Score: {silhouette_avg_reduced}')
calinski_harabasz_avg_reduced = calinski_harabasz_score(embeddings, reduced_clusters)
print(f'Reduced Calinski-Harabasz Index: {calinski_harabasz_avg_reduced}')
davies_bouldin_avg_reduced = davies_bouldin_score(embeddings, reduced_clusters)
print(f'Reduced Davies-Bouldin Index: {davies_bouldin_avg_reduced}')


# Finding closest posts to centroids
closest_posts = {i: [] for i in range(k)}
for i, centroid in enumerate(centroids):
    distances = np.linalg.norm(reduced_embeddings - centroid, axis=1)
    nearest_posts_indices = np.argsort(distances)[:10]
    closest_posts[i] = post_ids[nearest_posts_indices].values.tolist()

# Save to file
results_df = pd.DataFrame.from_dict(closest_posts, orient='index').reset_index()
results_df.columns = ['Centroid'] + [f'Closest_Post_{i+1}' for i in range(10)]
results_df.to_csv('64centroids_and_closest_posts.csv', index=False)

print("Centroids and their closest posts have been saved to 'centroids_and_closest_posts.csv'.")

# umap
nc_for_vis = 2
vis_embeddings = umap.UMAP(n_components=nc_for_vis, random_state=rs).fit_transform(embeddings)

# visual
plt.figure(figsize=(10, 8))
plt.scatter(vis_embeddings[:, 0], vis_embeddings[:, 1], c=reduced_clusters, alpha=0.4, cmap='tab20', s=10)
plt.colorbar(ticks=range(k))
plt.clim(-0.5, k - 0.5)
plt.savefig(fname='64reduced.pdf', format='pdf')
