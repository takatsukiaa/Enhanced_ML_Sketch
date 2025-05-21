import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# --- 1. 讀取資料 ---
# training_10.csv, testing_10.csv 是 10% 的資料train, 90% 的資料test
# training_50.csv, testing_50.csv 是 50% 的資料train, 50% 的資料test
train_df = pd.read_csv("~/ML-Sketch/Python/TopK/training.csv")
test_df = pd.read_csv("~/ML-Sketch/Python/TopK/testing.csv")

## --- 2. 特徵工程：加 log1p + ratio ---
for df in [train_df, test_df]:
    # log1p 特徵
    for col in ['initial_value', 'current_value', 'difference', 'c1', 'c2', 'c3', 'c4']:
        df[f'{col}_log'] = np.log1p(df[col])

    # 比例特徵
    df['delta_ratio'] = df['difference'] / (df['initial_value'] + 1)

    csum = df[['c1', 'c2', 'c3', 'c4']].sum(axis=1) + 1
    for c in ['c1', 'c2', 'c3', 'c4']:
        df[f'{c}_ratio'] = df[c] / csum

# --- 3. 特徵與標籤分離 ---
features = [
    'delta_ratio',
    'c1_ratio', 'c2_ratio', 'c3_ratio', 'c4_ratio',
    'c1_log', 'c2_log', 'c3_log', 'c4_log',
    'initial_value_log', 'current_value_log', 'difference_log'
]

X_train = train_df[features]
y_train = train_df['label']
X_test = test_df[features]
y_test = test_df['label']

# --- 4. 選擇 Scaler（可切換） ---
scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer(output_distribution='uniform')  # 或 'normal'

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. 模型訓練（XGBoost）---
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- 6. 評估結果 ---
y_pred = model.predict(X_test_scaled)

print("✅ ACC:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# --- 7. 特徵重要性可視化 ---
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()