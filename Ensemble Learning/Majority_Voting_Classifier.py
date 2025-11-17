import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------- Function to evaluate metrics ----------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_macro": precision_score(y_test, y_pred, average="macro"),
        "Recall_macro": recall_score(y_test, y_pred, average="macro"),
        "F1_macro": f1_score(y_test, y_pred, average="macro")
    }
    return metrics

def main(random_state=42, test_size=0.25):

    # Load dataset
    data = load_wine()
    X, y = data.data, data.target

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    # Base learners
    clf_lr  = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=42)
    clf_dt  = DecisionTreeClassifier(random_state=42)
    clf_rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_knn = KNeighborsClassifier(n_neighbors=5)

    clf_svm_soft = SVC(kernel="rbf", probability=True, random_state=42)   # For soft voting
    clf_svm_hard = SVC(kernel="rbf", random_state=42)                    # For hard voting

    base_learners = [
        ("LR", clf_lr),
        ("DT", clf_dt),
        ("RF", clf_rf),
        ("KNN", clf_knn),
        ("SVM", clf_svm_soft)
    ]

    # Evaluate individuals
    results = []
    print("\n\n=================== INDIVIDUAL CLASSIFIER PREDICTIONS ===================\n")
    
    for name, model in base_learners:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n{name} predictions on test set:\n{preds}")
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    # ---------------- HARD VOTING ----------------
    eclf_hard = VotingClassifier(
        estimators=[("LR", clf_lr), ("DT", clf_dt), ("RF", clf_rf),
                    ("KNN", clf_knn), ("SVM", clf_svm_hard)],
        voting="hard"
    )
    eclf_hard.fit(X_train, y_train)
    hard_preds = eclf_hard.predict(X_test)

    print("\n\n=================== HARD VOTING PREDICTIONS ===================\n")
    print(hard_preds)

    results.append(evaluate_model("Voting (HARD)", eclf_hard, X_train, X_test, y_train, y_test))


    # ---------------- SOFT VOTING ----------------
    eclf_soft = VotingClassifier(
        estimators=[("LR", clf_lr), ("DT", clf_dt), ("RF", clf_rf),
                    ("KNN", clf_knn), ("SVM", clf_svm_soft)],
        voting="soft"
    )
    eclf_soft.fit(X_train, y_train)
    soft_preds = eclf_soft.predict(X_test)

    print("\n\n=================== SOFT VOTING PREDICTIONS ===================\n")
    print(soft_preds)

    results.append(evaluate_model("Voting (SOFT)", eclf_soft, X_train, X_test, y_train, y_test))


    # Final results table
    df_results = pd.DataFrame(results).set_index("Model").round(4)
    print("\n\n=================== FINAL COMPARISON TABLE ===================\n")
    print(df_results)

    return df_results



if __name__ == "__main__":
    main()
