import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Scaling 'Amount'
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].to_numpy().reshape(-1, 1))
    
    # Normalizing 'Time'
    time = df['Time']
    df['Time'] = (time - time.min()) / (time.max() - time.min())
    
    return df

def apply_resampling_techniques(x, y, method='smote'):
    if method == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        x_resampled, y_resampled = rus.fit_resample(x, y)
    elif method == 'oversample':
        ros = RandomOverSampler(random_state=42)
        x_resampled, y_resampled = ros.fit_resample(x, y)
    elif method == 'smote':
        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(x, y)
    else:
        raise ValueError("Invalid resampling method")
    
    return x_resampled, y_resampled
