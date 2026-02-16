import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier



# -------------------- MODEL BUILD --------------------
def model_build(df):

    X = df.drop(columns=["Addicted", "addiction_lavel"], errors='ignore')
    y = df["Addicted"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.2
    )

   # Preprocessing (handle categorical + scaling)
    preprocessor = data_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False)
    }

    return models, X_train, X_test, y_train, y_test