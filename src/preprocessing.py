import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class WaterDataPreprocessor:
    """Handles cleaning and transforming water quality datasets."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols = ['pH', 'TDS', 'Turbidity', 'DO', 'BOD', 'Nitrates']

    def clean_data(self, df):
        """Basic data cleaning: handling missing values and duplicates."""
        # Drop duplicates
        df = df.drop_duplicates()
        
        # Fill missing numeric values with median
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def standardize_parameters(self, df):
        """Standardizes numeric features using StandardScaler."""
        available_cols = [c for c in self.numeric_cols if c in df.columns]
        if not available_cols:
            return df
            
        df_scaled = df.copy()
        df_scaled[available_cols] = self.scaler.fit_transform(df[available_cols])
        return df_scaled

    def calculate_custom_wqi(self, df):
        """
        Calculates a simplified Water Quality Index based on BIS standards.
        (Example formula for demonstration)
        """
        if not all(col in df.columns for col in ['pH', 'DO', 'BOD']):
            return df
            
        # Simplified WQI calculation: 
        # pH weight: 0.2, DO weight: 0.4, BOD weight: 0.4
        # (Relative to ideal values)
        w_ph = 0.2; w_do = 0.4; w_bod = 0.4
        
        # Normalize features to 0-100 scale (100 is best)
        # pH ideal 7.0, DO ideal > 6, BOD ideal < 3
        q_ph = 100 - abs(df['pH'] - 7.0) * 20
        q_do = (df['DO'] / 8.0) * 100
        q_bod = 100 - (df['BOD'] / 10.0) * 100
        
        df['Calculated_WQI'] = (q_ph * w_ph + q_do * w_do + q_bod * w_bod).clip(0, 100)
        return df
