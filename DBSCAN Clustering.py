# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.image as mpimg
import math

# --- 2. Define Paths ---
data_path = "C:/Users/massa/Desktop/projet passec/Data/PASEC2019_GRADE6_TREAT_ANALYSIS.dta"
save_path = "C:/Users/massa/Desktop/projet passec/Figure python/par_pays/"
os.makedirs(save_path, exist_ok=True)

# --- 3. Load Data ---
data = pd.read_stata(data_path)

# --- 4. Variables selection ---
features = [
    'math_score', 'reading_score',
    'ses_index', 'electricity_home',
    'books_26_100', 'hungry_in_class',
    'eats_lunch_school', 'school_infra'
]

# --- 5. Get list of countries ---
countries = data['PAYS'].dropna().unique()
countries.sort()

# --- 6. Initialize storage ---
summary = []

# --- 7. Start Loop Over Countries ---
for country in countries:
    print(f"\n=== Country: {country} ===")
    df_country = data[data['PAYS'] == country]
    X = df_country[features].dropna()

    if X.shape[0] < 50:
        print("Skipping (not enough students)")
        continue

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN
    eps_value = 0.8
    min_samples = 10
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN clusters: {n_clusters_dbscan}")

    # Fallback if necessary
    if 1 < n_clusters_dbscan <= 10:
        final_labels = dbscan_labels
        method = "DBSCAN"
    else:
        kmeans = KMeans(n_clusters=6, random_state=42)
        final_labels = kmeans.fit_predict(X_scaled)
        method = "KMeans"

    # PCA for 2D Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='plasma', s=10)
    plt.title(f'{country} - {method} Clustering (PCA)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + f"pca_{country}.png", dpi=300)
    plt.close()

    # Profile Clusters
    mask = final_labels != -1
    profile_df = pd.DataFrame(X[mask], columns=features)
    profile_df['Cluster'] = final_labels[mask]
    cluster_profiles = profile_df.groupby('Cluster').mean()

    # Save summary
    summary.append({
        'Country': country,
        'n_clusters': len(cluster_profiles),
        'Method': method,
        'Silhouette': silhouette_score(X_scaled[mask], final_labels[mask]) if len(set(final_labels[mask])) > 1 else np.nan
    })

    # Radar charts for each cluster
    normalized_profiles = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

    for cluster_id, row in normalized_profiles.iterrows():
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        values = row.tolist()
        values += values[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='teal', alpha=0.3)
        ax.plot(angles, values, color='teal', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=8)
        ax.set_title(f'{country} - Cluster {cluster_id}', y=1.1)
        plt.tight_layout()
        plt.savefig(save_path + f"radar_{country}_cluster_{cluster_id}.png", dpi=300)
        plt.close()

# --- 8. Save Summary Table ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv(save_path + "summary_by_country.csv", index=False)

print("\n✅ Finished clustering by country.")
print(summary_df)

# --- 9. Grid of PCA Plots, with 7 countries maximum per image ---
countries_for_grid = [f.split("_")[1].replace(".png", "") for f in os.listdir(save_path) if f.startswith("pca_")]
countries_for_grid = sorted(set(countries_for_grid))

n_countries_per_image = 7
n_total = len(countries_for_grid)
n_images = math.ceil(n_total / n_countries_per_image)

print(f"Creating {n_images} PCA grid images...")

for image_num in range(n_images):
    start_idx = image_num * n_countries_per_image
    end_idx = min((image_num + 1) * n_countries_per_image, n_total)
    selected_countries = countries_for_grid[start_idx:end_idx]

    n_cols = 3
    n_rows = math.ceil(len(selected_countries) / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    for idx, country in enumerate(selected_countries):
        img_path = save_path + f"pca_{country}.png"
        row, col = divmod(idx, n_cols)

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            axs[row, col].set_title(country)
        else:
            axs[row, col].axis('off')

    # Supprimer cases vides
    for idx in range(len(selected_countries), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path + f"pca_grid_part_{image_num+1}.png", dpi=300)
    plt.show()

    print(f"✅ Saved PCA grid part {image_num+1}")

import glob
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Chercher tous les fichiers radar
radar_files = glob.glob(os.path.join(save_path, "radar_*_cluster_*.png"))
print(f"Total radar charts found: {len(radar_files)}")

# 2. Organiser les fichiers par cluster ID
cluster_dict = {}

for file in radar_files:
    filename = os.path.basename(file)
    parts = filename.replace('.png', '').split('_')
    # Filename format: radar_COUNTRY_CLUSTERID.png
    cluster_id = parts[-1]

    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []

    cluster_dict[cluster_id].append(file)

# 3. Pour chaque cluster_id, créer plusieurs images avec 7 radars maximum par image

for cluster_id, files in cluster_dict.items():
    files.sort()  # Tri alphabétique

    n_files = len(files)
    n_radars_per_image = 7
    n_images = math.ceil(n_files / n_radars_per_image)

    print(f"\n📦 Cluster {cluster_id} - {n_files} radars to split into {n_images} images.")

    for image_num in range(n_images):
        start_idx = image_num * n_radars_per_image
        end_idx = min((image_num + 1) * n_radars_per_image, n_files)
        selected_files = files[start_idx:end_idx]

        n_cols = 3
        n_rows = math.ceil(len(selected_files) / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        for idx, img_path in enumerate(selected_files):
            row, col = divmod(idx, n_cols)
            img = mpimg.imread(img_path)
            axs[row, col].imshow(img)
            axs[row, col].axis('off')

            title = os.path.basename(img_path).replace('radar_', '').replace('.png', '').replace('_', ' ')
            axs[row, col].set_title(title, fontsize=8)

        # Supprimer les cases vides
        for idx in range(len(selected_files), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(save_path + f"radar_grid_cluster_{cluster_id}_part_{image_num+1}.png", dpi=300)
        plt.show()

        print(f"✅ Saved radar grid for Cluster {cluster_id} - part {image_num+1}")

