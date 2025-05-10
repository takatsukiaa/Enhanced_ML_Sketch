import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# --- 1. 讀取訓練與測試資料 ---
train_df = pd.read_csv('training.csv')
test_df = pd.read_csv('testing.csv')

# --- 2. 特徵與標籤拆分 ---
features = ['initial_value', 'current_value', 'difference', 'c1', 'c2', 'c3', 'c4']

X_train = train_df[features]
y_train = train_df['label']

X_test = test_df[features]
y_test = test_df['label']

# --- 3. 模型訓練（用 XGBoost）---
model = XGBClassifier(eval_metric='logloss', random_state=42)
# model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# --- 4. 預測與評估 ---
y_pred = model.predict(X_test)

print("ACC:", accuracy_score(y_test, y_pred))
print("report:")
print(classification_report(y_test, y_pred))

# --- 5. 特徵重要性輸出 ---
feature_names = X_train.columns
importances = model.feature_importances_

for name, score in zip(feature_names, importances):
    print(f"{name}: {score:.4f}")
#分析0, 1分布
import pandas as pd

train = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")

print("Train label distribution:")
print(train['label'].value_counts())

print("\nTest label distribution:")
print(test['label'].value_counts())
# （選擇性）畫圖
# plt.figure(figsize=(8, 5))
# plt.barh(feature_names, importances)
# plt.xlabel("Feature Importance Score")
# plt.title("XGBoost Feature Importances")
# plt.show()