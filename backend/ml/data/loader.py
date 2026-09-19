import os
import pandas as pd
import numpy as np

# CMAPSS columns based on standard documentation
COLUMNS = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 
           's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
           's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# Features to drop (sensors that don't change or settings that don't add value for FD001)
# For FD001, settings are constant/noise, and some sensors (s1, s5, s10, s16, s18, s19) are flat.
FEATURES_TO_DROP = ['setting1', 'setting2', 'setting3', 's1', 's5', 's10', 's16', 's18', 's19']
# Or we can just use all sensors, but typically dropping flat ones helps. We will keep the 14 useful sensors.
USEFUL_SENSORS = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
FEATURE_COLS = USEFUL_SENSORS

MAX_RUL = 125

def load_cmapss_data(data_dir: str, dataset_id: str = "FD001"):
    """
    Loads CMAPSS data. Returns train_df, test_df, and true_rul_df.
    """
    train_file = os.path.join(data_dir, f"train_{dataset_id}.txt")
    test_file = os.path.join(data_dir, f"test_{dataset_id}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{dataset_id}.txt")
    
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=COLUMNS)
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=COLUMNS)
    
    # RUL file has one value per engine ID
    true_rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
    true_rul_df['id'] = true_rul_df.index + 1
    
    return train_df, test_df, true_rul_df

def add_rul_labels(train_df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """
    Calculates Remaining Useful Life (RUL) for training data.
    Caps RUL at max_rul to avoid penalizing models for early-life variability.
    """
    rul_df = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul_df.columns = ['id', 'max_cycle']
    
    df = train_df.merge(rul_df, on='id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    if max_rul is not None:
        df['RUL'] = df['RUL'].clip(upper=max_rul)
        
    df = df.drop('max_cycle', axis=1)
    return df

def generate_sequences(df: pd.DataFrame, sequence_length: int, feature_cols: list):
    """
    Generates time-series sequences of length `sequence_length` for each engine unit.
    Returns array of shape (num_samples, sequence_length, num_features).
    """
    data = []
    labels = []
    
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        
        # We need at least sequence_length rows to make a sequence
        if len(engine_data) >= sequence_length:
            feature_array = engine_data[feature_cols].values
            rul_array = engine_data['RUL'].values if 'RUL' in engine_data.columns else None
            
            for i in range(len(engine_data) - sequence_length + 1):
                data.append(feature_array[i:i + sequence_length])
                if rul_array is not None:
                    labels.append(rul_array[i + sequence_length - 1])
                    
    if labels:
        return np.array(data), np.array(labels)
    return np.array(data)

def generate_test_sequences(test_df: pd.DataFrame, sequence_length: int, feature_cols: list):
    """
    For the test set, we only predict RUL for the *last* available sequence of each engine.
    """
    data = []
    engine_ids = []
    
    for engine_id in test_df['id'].unique():
        engine_data = test_df[test_df['id'] == engine_id]
        
        # If engine has less data than sequence length, pad it (for simplicity here, we assume seq_len <= min test length, which is 31 for FD001)
        # We just take the last sequence_length rows
        if len(engine_data) >= sequence_length:
            feature_array = engine_data[feature_cols].values[-sequence_length:]
            data.append(feature_array)
            engine_ids.append(engine_id)
            
    return np.array(data), np.array(engine_ids)
