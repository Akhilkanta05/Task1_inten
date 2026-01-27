# ==============================================
# CUSTOMER SEGMENTATION PROJECT - COMPLETE CODE
# ==============================================

# 1️⃣ Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import plotly.express as px

st.title("Customer Segmentation Project")

# ----------------------------------------------
# 2️⃣ Load Dataset
# ----------------------------------------------

st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file for customer segmentation", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed. Using sample data for demonstration.")
    # Create sample data
    np.random.seed(42)  # For reproducibility
    data = {
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    df = pd.DataFrame(data)

st.write("First 5 Rows:")
st.dataframe(df.head())

st.write("Dataset Info:")
st.text(str(df.info()))

# ----------------------------------------------
# 3️⃣ Data Cleaning
# ----------------------------------------------

# Check missing values
st.write("Missing Values:")
st.write(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Rename columns (if needed)
df.columns = df.columns.str.strip()

# Encode Gender
if "Gender" in df.columns:
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

# ----------------------------------------------
# 4️⃣ Feature Engineering
# ----------------------------------------------

# Create Spending Efficiency feature
if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
    df["Spending_Efficiency"] = df["Spending Score (1-100)"] / df["Annual Income (k$)"]

st.write("After Feature Engineering:")
st.dataframe(df.head())

# ----------------------------------------------
# 5️⃣ Exploratory Data Analysis (EDA)
# ----------------------------------------------

sns.set_style("whitegrid")

# Age Distribution
plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
st.pyplot(plt)

# Income vs Spending
plt.figure()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    data=df
)
plt.title("Income vs Spending Score")
st.pyplot(plt)

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
st.pyplot(plt)

# ----------------------------------------------
# 6️⃣ Feature Scaling
# ----------------------------------------------

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

if "Spending_Efficiency" in df.columns:
    features.append("Spending_Efficiency")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# ----------------------------------------------
# 7️⃣ Elbow Method
# ----------------------------------------------

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
st.pyplot(plt)

# ----------------------------------------------
# 8️⃣ Apply KMeans (Choose optimal K, assume 5)
# ----------------------------------------------

kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# Silhouette Score
score = silhouette_score(scaled_data, df["Cluster"])
st.write("Silhouette Score:", score)

# ----------------------------------------------
# 9️⃣ Cluster Visualization (2D)
# ----------------------------------------------

plt.figure()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    data=df
)
plt.title("Customer Segments")
st.pyplot(plt)

# ----------------------------------------------
# 🔟 PCA for 2D Visualization
# ----------------------------------------------

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]

plt.figure()
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    data=df
)
plt.title("PCA Cluster Visualization")
st.pyplot(plt)

# ----------------------------------------------
# 1️⃣1️⃣ 3D Interactive Plot (Plotly)
# ----------------------------------------------

fig = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster",
    title="3D Customer Segmentation"
)

st.plotly_chart(fig)

# ----------------------------------------------
# 1️⃣2️⃣ Cluster Summary
# ----------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean()
st.write("Cluster Summary:")
st.dataframe(cluster_summary)

# Save clustered dataset
df.to_csv("Customer_Segmented_Output.csv", index=False)

st.success("Project Completed Successfully 🚀")
