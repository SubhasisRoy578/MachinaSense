from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import os

class SensorScaler:
    def __init__(self, feature_cols):
        self.scaler = MinMaxScaler()
        self.feature_cols = feature_cols
        
    def fit(self, df: pd.DataFrame):
        self.scaler.fit(df[self.feature_cols])
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        
    def load(self, filepath: str):
        self.scaler = joblib.load(filepath)
