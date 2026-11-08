
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data.csv", sep=";")
df.columns = df.columns.str.strip()

X = df.drop("Target", axis=1)
y = df["Target"]

# --------------------------------------------------
# 2. Identify categorical and numerical features
# --------------------------------------------------

categorical_features = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Save the held-out test dataset for Streamlit
test_data = X_test.copy()
test_data["Target"] = y_test

test_data.to_csv("test_data.csv", sep=";", index=False)

print("Test dataset saved successfully!")

# --------------------------------------------------
# 4. Define preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# --------------------------------------------------
# 5. Define classifiers
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}

# --------------------------------------------------
# 6. Train, evaluate and save models
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

results = []

class_labels = sorted(y.unique())

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)

    # Complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Class probabilities for multiclass AUC
    y_prob = pipeline.predict_proba(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=class_labels
    )

    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    print("\nEvaluation Metrics:")
    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=class_labels))

    # Save complete pipeline
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(pipeline, os.path.join("model", filename))

# --------------------------------------------------
# 7. Save model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nAll models trained and saved successfully!")
