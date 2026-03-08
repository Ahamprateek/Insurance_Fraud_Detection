# trainingModel.py - ALL Stories 1.1-1.7 + PyCharm Clean (NO unresolved references)
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.model_scores_before = {}
        self.model_scores_after = {}
        self.feature_names = []

    def load_data(self):
        """Load training data from YOUR validated folder"""
        data_path = "Training_Raw_files_validated/fraudDetection_021119920_010222.csv"
        df = pd.read_csv(data_path)
        print(f"📊 Loaded data: {df.shape}")
        return df

    def preprocess_data(self, df):
        """Encoding Categorical Features + Scaling (COMPLETE)"""
        df_clean = df.copy()

        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        # Encode categorical columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            self.label_encoders[col] = le

        # Y/N mapping
        yn_map = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(yn_map)

        # Prepare X, y
        X = df_clean.iloc[:, :-1]  # All except last column
        y = pd.to_numeric(df_clean.iloc[:, -1], errors='coerce').fillna(0).astype(int)

        self.feature_names = X.columns.tolist()
        print(f"✅ Encoding + Scaling complete. Features: {len(X.columns)}")
        return X, y

    def train_base_models(self, x_train_scaled, y_train, x_test_scaled, y_test):
        """Stories 1.1-1.7: Train ALL 7 base models BEFORE tuning"""
        models_config = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),  # Story 1.1
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),  # Story 1.2
            'KNN': KNeighborsClassifier(n_neighbors=5),  # Activity 1.3
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),  # Story 1.4
            'NaiveBayes': GaussianNB(),  # Story 1.5
            'SVM': SVC(probability=True, random_state=42)  # Story 1.6
        }

        print("\n📊 BASE MODEL PERFORMANCE (BEFORE TUNING):")
        print("-" * 60)

        for model_name, model in models_config.items():
            model.fit(x_train_scaled, y_train)
            train_score = model.score(x_train_scaled, y_train)
            test_score = model.score(x_test_scaled, y_test)

            self.models[model_name] = model
            self.model_scores_before[model_name] = test_score

            print(f"{model_name:<18}: Test={test_score:.1%} | Train={train_score:.1%}")

    def hyperparameter_tuning(self, x_train_scaled, y_train, x_test_scaled, y_test):
        """Activity 2: Hyperparameter tuning (OPTIONAL)"""
        print("\n🔧 HYPERPARAMETER TUNING (Activity 2):")
        print("-" * 60)

        param_grids = {
            'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]},
            'RandomForest': {'n_estimators': [50, 100], 'max_depth': [10, None]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            'LogisticRegression': {'C': [0.1, 1, 10]},
            'SVM': {'C': [0.1, 1], 'kernel': ['rbf', 'linear']}
        }

        for model_name in ['DecisionTree', 'RandomForest', 'KNN', 'LogisticRegression', 'SVM']:
            if model_name in param_grids:
                print(f"Tuning {model_name}...")
                grid_search = GridSearchCV(
                    self.models[model_name],
                    param_grids[model_name],
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1
                )
                grid_search.fit(x_train_scaled, y_train)
                self.models[model_name] = grid_search.best_estimator_

                tuned_score = self.models[model_name].score(x_test_scaled, y_test)
                self.model_scores_after[model_name] = tuned_score
                print(f"  {model_name}: {tuned_score:.1%} (Best: {grid_search.best_params_})")

    def compare_models(self):
        """Story 1.7: Model Comparison"""
        print("\n" + "=" * 80)
        print("🏆 COMPLETE MODEL COMPARISON (Story 1.7)")
        print("=" * 80)
        print(f"{'Model':<20} {'Before':<10} {'After':<10} {'Gain':<8} {'Status'}")
        print("-" * 80)

        best_score = 0
        best_model = ""

        for model_name in self.models.keys():
            before_score = self.model_scores_before.get(model_name, 0)
            after_score = self.model_scores_after.get(model_name, before_score)
            gain = after_score - before_score
            status = "🏆 BEST" if after_score > best_score else "✓ GOOD"

            if after_score > best_score:
                best_score = after_score
                best_model = model_name

            gain_str = f"{gain:+.1%}"
            print(f"{model_name:<20} {before_score:<9.1%} {after_score:<9.1%} {gain_str:<7} {status}")

        print(f"\n🎉 WINNER: {best_model} ({best_score:.1%})")

    def run_training_pipeline(self):
        """Execute ALL Stories 1.1-1.7 + Activity 2"""
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")

        # Load data
        df = self.load_data()
        x_data, y_data = self.preprocess_data(df)

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )

        # Scale features (Scaling Story)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Train base models (Stories 1.1-1.6)
        self.train_base_models(x_train_scaled, y_train, x_test_scaled, y_test)

        # Hyperparameter tuning (Activity 2 - OPTIONAL)
        self.hyperparameter_tuning(x_train_scaled, y_train, x_test_scaled, y_test)

        # Model comparison (Story 1.7)
        self.compare_models()

        # Save everything
        os.makedirs("models", exist_ok=True)
        joblib.dump(self, 'models/complete_trainer.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("\n✅ SAVED: models/complete_trainer.pkl")
        print("🎉 ALL STORIES 1.1-1.7 + Activity 2 COMPLETE!")


def main():
    trainer = ModelTrainer()
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
