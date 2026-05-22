import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff

class WaterSafetyClassifier:
    """Unit III: Logistic Regression for Safe/Unsafe Classification."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.fpr = None
        self.tpr = None
        self.roc_auc = 0
        self.thresholds = None

    def fit(self, X, y):
        """Fits Logistic Regression and calculates ROC metrics."""
        self.model.fit(X, y)
        y_probs = self.model.predict_proba(X)[:, 1]
        
        self.fpr, self.tpr, self.thresholds = roc_curve(y, y_probs)
        self.roc_auc = auc(self.fpr, self.tpr)
        
        return y_probs

    def get_classification_report(self, X, y, threshold=0.5):
        """Returns classification report dict for a given threshold."""
        y_probs = self.model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        return classification_report(y, y_pred, output_dict=True, zero_division=0)

    def get_metrics(self, X, y, threshold=0.5):
        """Returns core metrics for UI display."""
        y_probs = self.model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return {"precision": precision, "recall": recall, "specificity": specificity, "cm": cm}

    def plot_plotly_confusion_matrix(self, X, y, threshold=0.5):
        """Plots high-fidelity Plotly confusion matrix."""
        y_probs = self.model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        # Reverse y-axis labels for standard top-to-bottom reading
        z = cm[::-1]
        y_labels = ['Unsafe (1)', 'Safe (0)']
        x_labels = ['Safe (0)', 'Unsafe (1)']
        
        fig = ff.create_annotated_heatmap(
            z, x=x_labels, y=y_labels, colorscale='Teal', showscale=True
        )
        fig.update_layout(
            title="Confusion Matrix Heatmap",
            title_font=dict(color='#f1f5f9', family="Outfit", size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Predicted Label", title_font=dict(color='#94a3b8', family="Outfit"), tickfont=dict(color='#cbd5e1', family="Outfit")),
            yaxis=dict(title="True Label", title_font=dict(color='#94a3b8', family="Outfit"), tickfont=dict(color='#cbd5e1', family="Outfit")),
            margin=dict(l=60, r=20, t=60, b=40)
        )
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 20
            fig.layout.annotations[i].font.family = "Outfit"
            fig.layout.annotations[i].font.color = "white"
        return fig

    def plot_plotly_roc(self, current_threshold=0.5):
        """Generates an interactive Plotly ROC curve with a moving threshold marker."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.fpr, y=self.tpr,
            mode='lines',
            name=f'ROC curve (AUC = {self.roc_auc:.3f})',
            line=dict(color='#f59e0b', width=3),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guessing',
            line=dict(color='rgba(255,255,255,0.3)', width=2, dash='dash'),
            hoverinfo='skip'
        ))
        
        if self.thresholds is not None:
            idx = (np.abs(self.thresholds - current_threshold)).argmin()
            curr_fpr = self.fpr[idx]
            curr_tpr = self.tpr[idx]
            
            fig.add_trace(go.Scatter(
                x=[curr_fpr], y=[curr_tpr],
                mode='markers+text',
                marker=dict(color='#38bdf8', size=14, line=dict(color='#fff', width=2)),
                name=f'Threshold: {current_threshold:.2f}',
                text=[f"Cutoff: {current_threshold:.2f}"],
                textposition="bottom right",
                textfont=dict(color='#38bdf8', family="Outfit", size=13),
                hovertemplate="Threshold: %{text}<br>TPR (Recall): %{y:.3f}<br>FPR: %{x:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            title_font=dict(color='#f1f5f9', family="Outfit", size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="False Positive Rate (1 - Specificity)", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit")),
            yaxis=dict(title="True Positive Rate (Sensitivity)", title_font=dict(color='#94a3b8', family="Outfit"), gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#cbd5e1', family="Outfit")),
            legend=dict(font=dict(color='#cbd5e1', family="Outfit"), bgcolor='rgba(15,23,42,0.6)'),
            margin=dict(l=60, r=40, t=60, b=60)
        )
        return fig
