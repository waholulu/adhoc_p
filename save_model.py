# pip install xgboost scikit-learn

import os
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1) Load public data
data = load_iris()
X, y = data.data, data.target
target_names = data.target_names

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Train an XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)
model.fit(X_train, y_train)

# Quick sanity check
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")

# 4) Save in UBJSON format
path = "iris_xgb.ubj"
model.save_model(path)
print(f"Saved model to: {os.path.abspath(path)}")

# 5) Load model and make a simple prediction
loaded = xgb.XGBClassifier()  # fresh wrapper; config/weights come from file
loaded.load_model(path)

sample = X_test[:1]
true_label = y_test[0]
pred_label = loaded.predict(sample)[0]
proba = loaded.predict_proba(sample)[0]

print("Sample features:", sample[0].tolist())
print(f"True label: {true_label} ({target_names[true_label]})")
print(f"Predicted label: {pred_label} ({target_names[pred_label]})")
print("Class probabilities:")
for i, p in enumerate(proba):
    print(f"  {i} ({target_names[i]}): {p:.3f}")
