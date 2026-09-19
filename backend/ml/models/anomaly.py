from sklearn.ensemble import IsolationForest
import joblib
import os
import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, contamination='auto', random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X):
        self.model.fit(X)
        
    def score_samples(self, X):
        """
        Returns an anomaly score. 
        Isolation Forest score_samples returns opposite of anomaly score (lower means more anomalous).
        We invert it so higher score = more anomalous.
        """
        scores = self.model.score_samples(X)
        # Invert to make higher = more anomalous. Typical scores are around [-0.5, 0.5]
        # We will normalize this later or just return the inverted score.
        return -scores
        
    def predict(self, X):
        """
        Returns 1 for normal, -1 for anomaly.
        """
        return self.model.predict(X)

    def evaluate_severity(self, anomaly_score, threshold_high=0.6, threshold_critical=0.75):
        """
        Maps a normalized anomaly score [0, 1] to severity.
        Since we inverted the score, we can pass normalized values here.
        """
        if anomaly_score > threshold_critical:
            return 'critical'
        elif anomaly_score > threshold_high:
            return 'high'
        elif anomaly_score > 0.5:
            return 'medium'
        return 'healthy'

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)
