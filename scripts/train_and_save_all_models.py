"""
Drug Court Decision Support System (DCDSS)
Train and Export All 5 ML Architectures + Counterfactual Random Forest Model
Based on the exact training pipelines in the project notebooks.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "nijdataset.xlsx"
MODEL_DIR = BASE_DIR / "Drug_Court_Decision_Support_System" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"[*] Loading NIJ dataset from: {DATA_PATH} ...")
# Read dataset
df = pd.read_excel(DATA_PATH)
print(f"[*] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

target_col = "Recidivism_Within_3years"
df[target_col] = df[target_col].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

# Static Features definition
categorical_features = [
    "Gender", "Race", "Age_at_Release", "Gang_Affiliated", "Supervision_Level_First",
    "Education_Level", "Dependents", "Prison_Offense", "Prison_Years",
    "Prior_Arrest_Episodes_Felony", "Prior_Arrest_Episodes_Misd", "Prior_Arrest_Episodes_Violent",
    "Prior_Arrest_Episodes_Property", "Prior_Arrest_Episodes_Drug", "Prior_Arrest_Episodes_DVCharges",
    "Prior_Arrest_Episodes_GunCharges", "Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd",
    "Prior_Conviction_Episodes_Viol", "Prior_Conviction_Episodes_Prop", "Prior_Conviction_Episodes_Drug",
    "Prior_Revocations_Parole", "Prior_Revocations_Probation", "Condition_MH_SA",
    "Condition_Cog_Ed", "Condition_Other"
]

numerical_features = ["Residence_PUMA", "Supervision_Risk_Score_First"]
all_features = categorical_features + numerical_features

# Intervention features for counterfactual simulation
intervention_features = [
    "Program_Attendances", "Program_UnexcusedAbsences", "Percent_Days_Employed",
    "Jobs_Per_Year", "DrugTests_THC_Positive", "DrugTests_Cocaine_Positive",
    "Violations_ElectronicMonitoring", "Violations_Instruction", "Violations_FailToReport"
]

# Ensure clean string representation for categorical
for col in categorical_features:
    df[col] = df[col].astype(str).fillna("Missing")

# Ensure numeric for numerical
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Sample 8,000 cases for fast, high-accuracy training if full is large
sample_size = min(8000, len(df))
train_df = df.sample(n=sample_size, random_state=42)

X = train_df[all_features].copy()
y = train_df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[*] Train partition: {len(X_train)}, Test partition: {len(X_test)}")

# Build Preprocessor
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)

print("[*] Fitting preprocessor ...")
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Save Preprocessor
preprocessor_path = MODEL_DIR / "preprocessor.joblib"
joblib.dump(preprocessor, preprocessor_path)
print(f"[+] Saved preprocessor to: {preprocessor_path}")

# Initialize 5 Models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, penalty="l2", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1),
    "Support Vector Machine": CalibratedClassifierCV(LinearSVC(C=0.1, random_state=42, max_iter=2000), cv=3),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, eval_metric="logloss", random_state=42)
}

metadata = {}

for name, clf in models.items():
    print(f"[*] Training {name} ...")
    clf.fit(X_train_trans, y_train)
    
    # Save Model
    clean_name = name.lower().replace(" ", "_")
    model_file = MODEL_DIR / f"{clean_name}_model.joblib"
    joblib.dump(clf, model_file)
    print(f"[+] Saved {name} model to {model_file.name}")
    
    # Evaluate
    y_pred = clf.predict(X_test_trans)
    y_prob = clf.predict_proba(X_test_trans)[:, 1]
    
    acc = round(float(accuracy_score(y_test, y_pred)) * 100, 2)
    auc = round(float(roc_auc_score(y_test, y_prob)), 3)
    prec = round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2)
    rec = round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2)
    brier = round(float(brier_score_loss(y_test, y_prob)), 3)
    
    metadata[name] = {
        "name": name,
        "filename": model_file.name,
        "accuracy": acc,
        "rocAuc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier": brier,
        "badge": "Recommended" if name == "XGBoost" else ("Counterfactual Base" if name == "Random Forest" else "Standard")
    }
    print(f"    -> Acc: {acc}%, AUC: {auc}, Prec: {prec}%, Rec: {rec}%, F1: {f1}%, Brier: {brier}")

# ----------------- TRAIN COUNTERFACTUAL SIMULATION MODEL -----------------
print("[*] Training Counterfactual Simulation Model (Random Forest) ...")
sim_features = all_features + intervention_features

for col in intervention_features:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)

X_sim = train_df[sim_features].copy()
y_sim = train_df[target_col].copy()

sim_num_features = numerical_features + intervention_features

sim_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, sim_num_features)
    ]
)

sim_clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
sim_pipeline = Pipeline(steps=[
    ("preprocessor", sim_preprocessor),
    ("classifier", sim_clf)
])

sim_pipeline.fit(X_sim, y_sim)
sim_model_path = MODEL_DIR / "counterfactual_simulation_model.joblib"
joblib.dump(sim_pipeline, sim_model_path)
print(f"[+] Saved counterfactual simulation model to: {sim_model_path.name}")

# Save metadata json
metadata_path = MODEL_DIR / "models_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"[+] Saved metadata to: {metadata_path.name}")
print("[✓] ALL 5 MODELS + SIMULATION PIPELINE TRAINED AND EXPORTED SUCCESSFULLY!")

