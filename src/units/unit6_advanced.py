import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

class AdvancedAnalytics:
    """Unit VI: Hierarchical + DBSCAN + PCA + AI Alerts."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.dbscan = DBSCAN(eps=1.5, min_samples=3)

    def run_pca(self, X):
        """Reduces feature space to 2D for visualization."""
        X_scaled = self.scaler.fit_transform(X)
        components = self.pca.fit_transform(X_scaled)
        return pd.DataFrame(components, columns=['PC1', 'PC2'])

    def detect_outliers(self, X):
        """Uses DBSCAN to find anomalous cities/districts."""
        X_scaled = self.scaler.fit_transform(X)
        outliers = self.dbscan.fit_predict(X_scaled)
        return outliers

    def plot_plotly_dendrogram(self, X, labels=None):
        """Generates interactive Plotly Hierarchical Clustering Dendrogram."""
        X_scaled = self.scaler.fit_transform(X)
        
        # We use figure_factory to create a native Plotly dendrogram
        fig = ff.create_dendrogram(X_scaled, labels=labels, color_threshold=5.0)
        
        fig.update_layout(
            width=800,
            height=500,
            title="Hierarchical Risk Proximity Dendrogram",
            title_font=dict(color='#f1f5f9', family="Outfit", size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="District ID", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit")),
            yaxis=dict(title="Ward Linkage Distance", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit")),
            margin=dict(l=60, r=20, t=60, b=80)
        )
        # Update lines color for dark theme visibility
        for trace in fig.data:
            if trace.line.color == 'rgb(0,0,0)':
                trace.line.color = '#38bdf8' # Cyan
            elif trace.line.color == 'rgb(255,0,0)':
                trace.line.color = '#f59e0b' # Amber
            elif trace.line.color == 'rgb(0,255,0)':
                trace.line.color = '#10b981' # Emerald
            elif trace.line.color == 'rgb(0,0,255)':
                trace.line.color = '#8b5cf6' # Violet
        return fig

    def generate_ai_alert_stream(self, district_data):
        """Generates an advisory string for the AI typewriter simulation."""
        wqi = district_data.get('WQI', 'N/A')
        complaints = district_data.get('Complaints', 0)
        district_name = district_data.get('name', district_data.get('District', 'Unknown'))
        
        if isinstance(wqi, (int, float)) and wqi < 50:
            status = 'CRITICAL RISK DETECTED'
            action = 'IMMEDIATE MITIGATION REQUIRED'
            details = (f"> Detected systemic watershed degradation in {district_name}.\n"
                       f"> {complaints} civilian complaints registered in the last 48 hours.\n\n"
                       f"**ACTION PROTOCOL:**\n"
                       f"- Dispatch rapid-response hazard teams to municipal pipelines.\n"
                       f"- Increase standard chlorine dosing by 1.25x immediately.\n"
                       f"- Issue 'Boil Water Advisory' to local health portals.")
        else:
            status = 'STABLE WATER QUALITY'
            action = 'STANDARD MONITORING'
            details = (f"> Routine checks indicate {district_name} is within safe thresholds (WQI: {wqi}).\n"
                       f"> Only {complaints} minor maintenance requests registered.\n\n"
                       f"**ACTION PROTOCOL:**\n"
                       f"- Continue automated daily telemetry scraping.\n"
                       f"- No chemical dosage adjustments necessary.")
            
        alert_template = f"""
## 🤖 JalRakshak Municipal Command AI

**SYSTEM STATUS**: `{status}`
**TARGET SECTOR**: {district_name}
**DIRECTIVE**: {action}

---

{details}

> *End of automated diagnostic stream.*
        """
        return alert_template
