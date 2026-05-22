import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

class WQIParameterModel:
    """Unit II: Multiple Linear Regression for parameter-based prediction."""
    def __init__(self):
        self.model = LinearRegression()
        self.vif_data = None
        self.rfe_ranking = None

    def preprocess_categorical(self, df):
        """Encodes categorical features like Source and State."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df

    def calculate_vif(self, X):
        """Calculates Variance Inflation Factor for multicollinearity detection."""
        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        if X.shape[1] <= 1:
            vif["VIF"] = [1.0] * X.shape[1]
        else:
            vifs = []
            for i in range(len(X.columns)):
                try:
                    val = variance_inflation_factor(X.values, i)
                    if np.isinf(val) or np.isnan(val):
                        val = 9999.0
                    vifs.append(val)
                except Exception:
                    vifs.append(9999.0)
            vif["VIF"] = vifs
        self.vif_data = vif
        return vif

    def run_rfe(self, X, y, n_features=5):
        """Runs Recursive Feature Elimination to find top predictors."""
        n_features = min(n_features, X.shape[1])
        if n_features < 1:
            return pd.DataFrame(columns=['Feature', 'Selected', 'Ranking'])
        selector = RFE(self.model, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        self.rfe_ranking = pd.DataFrame({
            'Feature': X.columns,
            'Selected': selector.support_,
            'Ranking': selector.ranking_
        }).sort_values(by='Ranking')
        return self.rfe_ranking

    def fit(self, X, y):
        """Fits MLR and returns the summary (including Adjusted R²)."""
        X_with_const = sm.add_constant(X)
        model_sm = sm.OLS(y, X_with_const).fit()
        return model_sm

    def get_coefficient_stats(self, model_sm, alpha=0.05):
        """Extracts OLS coefficients and confidence intervals in a clean DataFrame."""
        if model_sm is None:
            return pd.DataFrame()
        
        params = model_sm.params
        bse = model_sm.bse
        tvalues = model_sm.tvalues
        pvalues = model_sm.pvalues
        conf = model_sm.conf_int(alpha=alpha)
        
        stats_df = pd.DataFrame({
            'Feature': params.index,
            'Coefficient': params.values,
            'Std_Err': bse.values,
            't_Stat': tvalues.values,
            'p_Value': pvalues.values,
            'Conf_Low': conf[0].values,
            'Conf_High': conf[1].values
        })
        
        stats_df['Significant'] = stats_df['p_Value'] < alpha
        return stats_df
