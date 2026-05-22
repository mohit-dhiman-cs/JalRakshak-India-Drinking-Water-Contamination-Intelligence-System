import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

class WQITrendModel:
    """Unit I: Simple Linear Regression for WQI Trend Analysis."""
    def __init__(self):
        self.model = LinearRegression()
        self.stats_model = None
        self.r2 = 0
        self.rmse = 0

    def fit(self, df):
        """Fits SLR and performs hypothesis testing."""
        df['Date_Ordinal'] = pd.to_datetime(df['Date']).apply(lambda x: x.toordinal())
        X = df[['Date_Ordinal']].values
        y = df['WQI'].values
        
        # sklearn fit for prediction
        self.model.fit(X, y)
        preds = self.model.predict(X)
        
        # statsmodels fit for hypothesis testing
        X_const = sm.add_constant(X)
        self.stats_model = sm.OLS(y, X_const).fit()
        
        self.r2 = r2_score(y, preds)
        self.rmse = np.sqrt(mean_squared_error(y, preds))
        
        return preds

    def get_summary(self):
        """Returns statistical summary including p-values for hypothesis testing."""
        if self.stats_model:
            return self.stats_model.summary()
        return "Model not fitted yet."

    def get_metrics(self):
        """Returns a dictionary of key statistical indicators for UI reporting."""
        if self.stats_model is None:
            return None
        return {
            'slope': self.stats_model.params[1],
            'p_value': self.stats_model.pvalues[1],
            't_stat': self.stats_model.tvalues[1],
            'conf_low': self.stats_model.conf_int()[1][0],
            'conf_high': self.stats_model.conf_int()[1][1],
            'r2': self.r2,
            'rmse': self.rmse
        }

    def plot_residuals(self, df, preds):
        """Unit I: Residual Analysis."""
        residuals = df['WQI'] - preds
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(preds, residuals)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title("Residual Plot (Check for Heteroscedasticity)")
        ax.set_xlabel("Predicted WQI")
        ax.set_ylabel("Residuals")
        return fig

    def plot_trend(self, df, preds, station_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['Date'], df['WQI'], color='blue', label='Actual WQI', alpha=0.6)
        ax.plot(df['Date'], preds, color='red', linewidth=2, label='Trend Line')
        ax.set_title(f"WQI Degradation Trend: {station_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("WQI")
        ax.legend()
        return fig

    def plot_t_distribution(self, alpha=0.05, tail='left', observed_t=None, df_val=None):
        """Generates an interactive Plotly visualization of the Student's t-distribution under H0."""
        import plotly.graph_objects as go
        from scipy import stats
        
        # 1. Determine t-statistic and degrees of freedom
        if observed_t is None:
            if self.stats_model is None:
                return None
            observed_t = self.stats_model.tvalues[1]
            df_val = self.stats_model.df_resid
        
        # 2. Calculate critical values
        if tail == 'left':
            t_crit_low = stats.t.ppf(alpha, df_val)
            t_crit_high = None
        elif tail == 'right':
            t_crit_low = None
            t_crit_high = stats.t.ppf(1 - alpha, df_val)
        else: # two-tailed
            t_crit_low = stats.t.ppf(alpha / 2, df_val)
            t_crit_high = stats.t.ppf(1 - alpha / 2, df_val)
            
        # 3. Generate curve coordinates
        limit = max(6.0, abs(observed_t) + 2.0)
        x = np.linspace(-limit, limit, 1000)
        y = stats.t.pdf(x, df_val)
        
        fig = go.Figure()
        
        # Plot complete t-distribution curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name="Null Hypothesis (H₀) Curve",
            line=dict(color='#38bdf8', width=2.5),
            hovertemplate="t-value: %{x:.3f}<br>Probability Density: %{y:.4f}<extra></extra>"
        ))
        
        # Shading Rejection Regions
        # Left tail rejection region
        if tail in ['left', 'two-tailed'] and t_crit_low is not None:
            x_left = x[x <= t_crit_low]
            y_left = y[x <= t_crit_low]
            fig.add_trace(go.Scatter(
                x=x_left, y=y_left,
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(239, 68, 68, 0.35)',
                name='Left Rejection Region',
                hovertemplate="Rejection Region (t < %{x:.3f})<extra></extra>"
            ))
            
        # Right tail rejection region
        if tail in ['right', 'two-tailed'] and t_crit_high is not None:
            x_right = x[x >= t_crit_high]
            y_right = y[x >= t_crit_high]
            fig.add_trace(go.Scatter(
                x=x_right, y=y_right,
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(239, 68, 68, 0.35)',
                name='Right Rejection Region',
                hovertemplate="Rejection Region (t > %{x:.3f})<extra></extra>"
            ))
            
        # Add observed t-statistic vertical line
        # Determine color: is it in the rejection region?
        is_rejected = False
        if tail == 'left' and observed_t <= t_crit_low:
            is_rejected = True
        elif tail == 'right' and observed_t >= t_crit_high:
            is_rejected = True
        elif tail == 'two-tailed' and (observed_t <= t_crit_low or observed_t >= t_crit_high):
            is_rejected = True
            
        line_color = '#ef4444' if is_rejected else '#10b981'
        status_text = "REJECT H₀" if is_rejected else "FAIL TO REJECT H₀"
        
        fig.add_vline(
            x=observed_t,
            line_width=3,
            line_dash="dash",
            line_color=line_color,
            annotation_text=f"Observed t: {observed_t:.3f} ({status_text})",
            annotation_position="top right",
            annotation_font=dict(color=line_color, size=13, family="Outfit")
        )
        
        # General layout styling to fit the premium dark glassmorphism look
        fig.update_layout(
            title={
                'text': f"Student's t-Distribution Probability Density Curve (df = {df_val})",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'family': 'Outfit', 'size': 18, 'color': '#f1f5f9'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title="t-Statistic Scale (Standard Deviations from 0)",
                title_font=dict(color='#94a3b8', family="Outfit"),
                gridcolor='rgba(255,255,255,0.05)',
                zerolinecolor='rgba(255,255,255,0.15)',
                tickfont=dict(color='#cbd5e1', family="Outfit")
            ),
            yaxis=dict(
                title="Probability Density",
                title_font=dict(color='#94a3b8', family="Outfit"),
                gridcolor='rgba(255,255,255,0.05)',
                zerolinecolor='rgba(255,255,255,0.15)',
                tickfont=dict(color='#cbd5e1', family="Outfit")
            ),
            legend=dict(
                font=dict(color='#cbd5e1', family="Outfit"),
                bgcolor='rgba(15,23,42,0.7)',
                bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1
            ),
            margin=dict(l=40, r=40, t=70, b=40)
        )
        
        return fig

