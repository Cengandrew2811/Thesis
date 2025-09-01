import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from matplotlib import cm
from patient import Patient


df = pd.read_excel("Copy of mmc1[66].xlsx")
exclude_ids = ["580.1", "613.3", "614.1"]
df["ID"] = df["ID"].astype(str)
df = df[~df["ID"].isin(exclude_ids)]
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df = df[df["Age"] >= 18]


required_columns = [
    'ID', 'Body weight', 'Height', 'Gender: 1, male; 2, female',
    'Age', 'hematocrit', 'serum creatinine'
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# === Extract per-patient data ===
patient_ids, weights, ages, gfr_values = [], [], [], []
for patient_id, group in df.groupby('ID'):
    try:
        weight = float(group.iloc[0]['Body weight'])
        age = float(group.iloc[0]['Age'])
    except:
        continue

    gfrs = []
    for _, row in group.iterrows():
        try:
            scr = float(row['serum creatinine']) * 0.0113
            hct = float(row['hematocrit']) / 100.0
            patient = Patient(
                row['Gender: 1, male; 2, female'],
                row['Body weight'],
                row['Height'],
                row['Age'],
                hct,
                scr
            )
            gfrs.append(patient.compute_gfr_ckd_epi())
        except:
            continue

    if gfrs:
        patient_ids.append(patient_id)
        weights.append(weight)
        ages.append(age)
        gfr_values.append(np.median(gfrs))

# === Build DataFrame ===
data = pd.DataFrame({
    'PatientID': patient_ids,
    'Weight': weights,
    'Age': ages,
    'GFR': gfr_values
})
X = data[['Weight', 'Age', 'GFR']]

# === Standardize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Elbow + Silhouette Score ===
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)', fontsize=16)
plt.ylabel('Inertia (Within-Cluster SSE)', fontsize=16)
plt.title('Elbow Method for Optimal K', fontsize=18)
plt.grid(True)
plt.xticks(K_range, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (K)', fontsize=16)
plt.ylabel('Silhouette Score', fontsize=16)
plt.title('Silhouette Score for KMeans Clustering', fontsize=18)
plt.grid(True)
plt.xticks(K_range, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# === Final Clustering ===
best_k = 5  # set based on your analysis of previous plots
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)
data['Cluster'] = labels

# === Cluster profile table in original units (Table R2) ===
from sklearn.metrics import silhouette_samples
import numpy as np
import pandas as pd

# back-transform centroids to original units
centers_scaled = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_scaled)  # columns are ['Weight','Age','GFR'] order

centers_df = pd.DataFrame(centers_orig, columns=['Weight','Age','GFR'])
centers_df.insert(0, 'Cluster', range(best_k))

# cluster sizes
size_by_cluster = pd.Series(labels).value_counts().sort_index()
centers_df['n'] = centers_df['Cluster'].map(size_by_cluster)

# eGFR category for centroid (for readability)
def egfr_cat(x):
    if x >= 90:
        return 'Normal (≥90)'
    elif x >= 60:
        return 'Mild impairment (60–89)'
    else:
        return 'Moderate–Severe (<60)'
centers_df['eGFR category'] = centers_df['GFR'].apply(egfr_cat)

# weight tertiles (based on your cohort)
q1, q2 = data['Weight'].quantile([1/3, 2/3]).values
def weight_tertile(w):
    return 'Low' if w < q1 else ('Mid' if w < q2 else 'High')
centers_df['Weight tertile'] = centers_df['Weight'].apply(weight_tertile)

# mean silhouette per cluster (quality diagnostic)
sil_vals = silhouette_samples(X_scaled, labels)
mean_sil = pd.Series(sil_vals).groupby(labels).mean()
centers_df['Mean silhouette s_k'] = centers_df['Cluster'].map(mean_sil)

# per-cluster SSE and % of total inertia (computed in scaled space to match kmeans.inertia_)
per_cluster_sse = []
for k in range(best_k):
    diffs = X_scaled[labels == k] - centers_scaled[k]
    sse = np.sum(np.sum(diffs**2, axis=1))
    per_cluster_sse.append(sse)
per_cluster_sse = np.array(per_cluster_sse)
centers_df['% inertia'] = (per_cluster_sse / kmeans.inertia_) * 100

# tidy types/rounding and column order
centers_df['Age'] = centers_df['Age'].round(1)
centers_df['Weight'] = centers_df['Weight'].round(1)
centers_df['GFR'] = centers_df['GFR'].round(1)
centers_df['Mean silhouette s_k'] = centers_df['Mean silhouette s_k'].round(2)
centers_df['% inertia'] = centers_df['% inertia'].round(1)

table_R2 = centers_df[['Cluster','n','Age','Weight','GFR',
                       'eGFR category','Weight tertile','Mean silhouette s_k','% inertia']]

print("\nTable R2. Cluster profiles mapped to clinical features (centroids in original units):\n")
print(table_R2.to_string(index=False))

# save for your appendix / SI
table_R2.to_csv("cluster_profiles.csv", index=False)
print("\n Saved: cluster_profiles.csv")

# Save
data.to_csv("patient_clusterss.csv", index=False)
print(f"Clustering complete with K={best_k}. Saved to 'patient_clusterss.csv'.")

# === Cluster 2D Visualization (discrete legend, bigger fonts) ===
import numpy as np
import matplotlib.pyplot as plt

unique_clusters = np.sort(data['Cluster'].unique())
palette = plt.cm.tab10(np.linspace(0, 1, max(10, best_k)))  # up to 10 distinct colors

fig, ax = plt.subplots(figsize=(8, 6))
for k in unique_clusters:
    sel = data['Cluster'] == k
    ax.scatter(
        data.loc[sel, 'Weight'],
        data.loc[sel, 'GFR'],
        s=80, edgecolors='k', linewidths=0.6,
        color=palette[k], label=f'Cluster {k}'
    )

ax.set_xlabel('Weight', fontsize=16)
ax.set_ylabel('GFR', fontsize=16)
ax.grid(True, alpha=0.4)
ax.legend(fontsize=10, frameon=True, ncol=3)
ax.tick_params(axis='both', labelsize=12)

plt.show()

# === Silhouette Analysis Plot ===
silhouette_vals = silhouette_samples(X_scaled, labels)
y_lower = 10
plt.figure(figsize=(10, 6))

for i in range(best_k):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / best_k)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # space between plots

plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
plt.xlabel("The silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title(f"Silhouette Plot for KMeans (K={best_k})")
plt.show()