# preprocessing.py - Encoding Categorical Features + Scaling
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def encode_categorical(self, df):
        """🔧 Encoding Categorical Features"""
        df_clean = df.copy()

        # Handle missing values
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

        # Encode categorical columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le

        # Y/N mapping
        yn_map = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(yn_map)

        return df_clean

    def scale_features(self, X_train, X_test):
        """📏 Feature Scaling"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def save_preprocessor(self):
        joblib.dump(self, 'models/preprocessor.pkl')
        print("✅ Preprocessor saved!")


# Usage
preprocessor = DataPreprocessor()
print("✅ Encoding + Scaling ready!")
