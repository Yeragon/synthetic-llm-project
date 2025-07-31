import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def tstr_evaluate(real_df, synthetic_df, target_col):
    """
    Train-on-synthetic, test-on-real (TSTR) classification evaluation
    """
    if target_col not in real_df.columns or target_col not in synthetic_df.columns:
        raise ValueError("Target column not found in both datasets")

    features = [col for col in synthetic_df.columns if col != target_col]

    # Align test and train features
    X_test = real_df[features]
    y_test = real_df[target_col]
    X_train = synthetic_df[features]
    y_train = synthetic_df[target_col]

    # Encode target if categorical
    if y_train.dtype == object:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # One-hot encode features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }