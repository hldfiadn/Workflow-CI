import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import mlflow
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from feature_engine.encoding import CountFrequencyEncoder

file_path = sys.argv[4] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MLProject/personality_dataset_preprocessing.csv"
)

data = pd.read_csv(file_path)

X = data.drop(columns=['Personality'])
y = data['Personality']

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 147
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.025025471015810354

model = Pipeline(steps=[
    ('preprocessor', 'passthrough'),
    ('classifier', XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate
    ))
])

# Split untuk evaluasi akhir dan logging
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

with mlflow.start_run():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Logging model dan parameternya
    signature = mlflow.models.signature.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        model, 
        "xgb_pipeline", 
        signature=signature, 
        input_example=X_test.iloc[:5]
    )

    mlflow.log_params(model.named_steps["classifier"].get_params())

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.savefig("conf_matrix.png")

    # Untuk MLflow (optional tapi disarankan)
    mlflow.log_artifact("conf_matrix.png")





