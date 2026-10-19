
import json
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_data

MODEL_DIR = Path("models")
FIGURE_DIR = Path("reports/figures")

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


def build_pipeline(estimator, X):
    categorical = [
        c for c in CATEGORICAL_COLUMNS if c in X.columns
    ]
    numerical = [
        c for c in X.columns if c not in categorical
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])


def main():
    df = load_data()

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-classification")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_model = None
    best_name = None
    best_auc = -1.0
    best_metrics = None

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator, X_train)

        with mlflow.start_run(run_name=name) as run:
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=[
                    "accuracy",
                    "precision",
                    "recall",
                    "roc_auc"
                ],
                error_score="raise"
            )

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_accuracy": float(
                    cv_results["test_accuracy"].mean()
                ),
                "cv_precision": float(
                    cv_results["test_precision"].mean()
                ),
                "cv_recall": float(
                    cv_results["test_recall"].mean()
                ),
                "cv_roc_auc": float(
                    cv_results["test_roc_auc"].mean()
                ),
                "test_accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "test_precision": float(
                    precision_score(
                        y_test, predictions, zero_division=0
                    )
                ),
                "test_recall": float(
                    recall_score(
                        y_test, predictions, zero_division=0
                    )
                ),
                "test_roc_auc": float(
                    roc_auc_score(y_test, probabilities)
                )
            }

            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(metrics)

            # Save confusion matrix
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions
            )
            plt.tight_layout()

            cm_path = FIGURE_DIR / (
                name.lower().replace(" ", "_") + "_confusion.png"
            )
            plt.savefig(cm_path, dpi=150)
            plt.close()

            # Save ROC curve
            RocCurveDisplay.from_predictions(
                y_test, probabilities
            )
            plt.tight_layout()

            roc_path = FIGURE_DIR / (
                name.lower().replace(" ", "_") + "_roc.png"
            )
            plt.savefig(roc_path, dpi=150)
            plt.close()

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            # Log reusable preprocessing + model pipeline
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle"
            )

            print(f"\nResults for {name}")
            print(json.dumps(metrics, indent=2))
            print("MLflow run ID:", run.info.run_id)

            # Select by cross-validation AUC, not test AUC
            if metrics["cv_roc_auc"] > best_auc:
                best_auc = metrics["cv_roc_auc"]
                best_model = pipeline
                best_name = name
                best_metrics = metrics

    # Save selected model and metadata
    joblib.dump(best_model, MODEL_DIR / "model.joblib")

    metadata = {
        "selected_model": best_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "cv_roc_auc": best_auc,
        "metrics": best_metrics,
        "features": list(X.columns)
    }

    (MODEL_DIR / "model_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nSelected model:", best_name)
    print("Saved model to models/model.joblib")


if __name__ == "__main__":
    main()
