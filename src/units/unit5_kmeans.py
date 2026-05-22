import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from random import sample
import plotly.graph_objects as go
import plotly.express as px

class DistrictRiskClusterer:
    """Unit V: K-Means Clustering for District Risk Segmentation."""
    def __init__(self, n_clusters=4):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.silhouette = 0

    def calculate_hopkins(self, X):
        """Calculates Hopkins Statistic to check for clustering tendency."""
        d = X.shape[1]
        n = len(X)
        m = int(0.1 * n)
        if m < 1:
            m = 1
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
        rand_X = np.random.uniform(X.min(axis=0), X.max(axis=0), (m, d))
        real_X = X.sample(m).values
        
        u_distances, _ = nbrs.kneighbors(rand_X, n_neighbors=1)
        w_distances, _ = nbrs.kneighbors(real_X, n_neighbors=2) 
        
        u_sum = np.sum(u_distances)
        w_sum = np.sum(w_distances[:, 1])
        
        hopkins_stat = u_sum / (u_sum + w_sum) if (u_sum + w_sum) > 0 else 0
        return hopkins_stat

    def fit(self, X):
        """Standardizes data and clusters districts."""
        X_scaled = self.scaler.fit_transform(X)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        if len(np.unique(clusters)) > 1:
            self.silhouette = silhouette_score(X_scaled, clusters)
            
        return clusters

    def plot_plotly_elbow(self, X, max_k=10):
        """Interactive Plotly Elbow Method Curve."""
        X_scaled = self.scaler.transform(X)
        distortions = []
        K = list(range(1, max_k + 1))
        for k in K:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            distortions.append(km.inertia_)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K, y=distortions,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#38bdf8', width=3),
            marker=dict(size=10, color='#f59e0b', line=dict(color='white', width=1)),
            hovertemplate="Clusters (k): %{x}<br>Distortion (Inertia): %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            title="Elbow Method for Optimal k",
            title_font=dict(color='#f1f5f9', family="Outfit", size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Number of Clusters (k)", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit"), dtick=1),
            yaxis=dict(title="Distortion (Inertia)", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit")),
            margin=dict(l=60, r=40, t=60, b=40)
        )
        return fig

    def plot_plotly_3d_scatter(self, df, clusters):
        """Generates a 3D scatter plot if specific features exist."""
        plot_df = df.copy()
        plot_df['Cluster'] = clusters.astype(str)
        
        # Pick 3 dimensions for the 3D plot
        cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) >= 3:
            x_col, y_col, z_col = cols[0], cols[1], cols[2]
            
            fig = px.scatter_3d(
                plot_df, x=x_col, y=y_col, z=z_col,
                color='Cluster',
                hover_name=plot_df.index if 'District' not in plot_df.columns else plot_df['District'],
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)', title_font=dict(color='#94a3b8', family="Outfit"), tickfont=dict(color='#cbd5e1', family="Outfit")),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)', title_font=dict(color='#94a3b8', family="Outfit"), tickfont=dict(color='#cbd5e1', family="Outfit")),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)', title_font=dict(color='#94a3b8', family="Outfit"), tickfont=dict(color='#cbd5e1', family="Outfit")),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(font=dict(color='#cbd5e1', family="Outfit"), bgcolor='rgba(15,23,42,0.6)', yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            return fig
        return go.Figure()

    def get_cluster_profiles(self, df, clusters):
        df_profile = df.copy()
        df_profile['Cluster'] = clusters
        profiles = df_profile.groupby('Cluster').mean(numeric_only=True)
        return profiles
