import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy import stats

def remove_outliers(X, threshold=3.0):
    
    z_scores = np.abs(stats.zscore(X))  # Calculate Z-scores
    m=np.nanmean(np.array(z_scores))
    z_scores_filled = z_scores.apply(lambda col: col.fillna(m), axis=0)

    threshold = 3  # Define a threshold for outlier removal
    return X[(z_scores_filled < threshold).all(axis=1)]

def balance_data(X_train, y_train):
    X_resampled, y_resampled = X_train.copy(), y_train.copy()
    X_resampled['state'] = y_resampled  # Add target column to features

    minority_class = X_resampled[X_resampled['state'] == 1]  # Assume 1 is the minority class
    majority_class = X_resampled[X_resampled['state'] == 0]  # Assume 0 is the majority class

    minority_upsampled = resample(minority_class, 
                                   replace=True,    # Sample with replacement
                                   n_samples=len(majority_class),  # Equal number of samples as majority class
                                   random_state=443)  # For reproducibility

    X_resampled = pd.concat([majority_class, minority_upsampled])
    y_resampled = X_resampled.pop('state')
    
    return X_resampled, y_resampled

def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled