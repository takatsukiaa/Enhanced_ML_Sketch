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
elephant_threshold = 2000
## --- 2. 特徵工程：加 log1p + ratio ---
for df in [train_df, test_df]:
    # log1p 特徵
    for col in [ 'c1', 'c2', 'c3', 'c4']:
        df[f'{col}_log'] = np.log1p(df[col])


    csum = df[['c1', 'c2', 'c3', 'c4']].sum(axis=1) + 1
    for c in ['c1', 'c2', 'c3', 'c4']:
        df[f'{c}_ratio'] = df[c] / csum
        # 統計特徵
    df['c_log_min'] = df[['c1_log', 'c2_log', 'c3_log', 'c4_log']].min(axis=1)
    df['c_max'] = df[['c1', 'c2', 'c3', 'c4']].max(axis=1)
    df['c_mean'] = df[['c1', 'c2', 'c3', 'c4']].mean(axis=1)
    df['c_std'] = df[['c1', 'c2', 'c3', 'c4']].std(axis=1)
    # elephant flag
    df['elephant'] = 0
    df.loc[df[['c1', 'c2', 'c3', 'c4']].min(axis=1) > elephant_threshold, 'elephant'] = 1

# 3. 特徵欄位設定
features = [
    'c1_log', 'c2_log', 'c3_log', 'c4_log',
    'c1_ratio', 'c2_ratio', 'c3_ratio', 'c4_ratio',
    # 'c_log_min', 
    # 'c_max',
    # 'c_mean',
    'c_std',
    # 'elephant'
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

# 計算 class weight
neg, pos = np.bincount(y_train)
scale = neg / pos
# --- 5. 模型訓練（XGBoost）---
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='error',
    random_state=42,
    scale_pos_weight=scale*2
)
model.fit(X_train_scaled, y_train)

# --- 6. 評估結果 ---
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # 機率為正類（label=1）

threshold = 0.999 # 可以改
y_pred = (y_pred_proba >= threshold).astype(int)

k = 250
top_k_indices = np.argsort(y_pred_proba)[-k:][::-1]
top_k_true = y_test[top_k_indices]
precision_at_k = np.sum(top_k_true) / k

print("✅ ACC:", accuracy_score(y_test, y_pred))

# print("\n✅ Classification Report:")
# print(classification_report(y_test, y_pred, digits=3))

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


#TPR, TNR, FPR, FNR
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
cm_reversed = cm[::-1, ::-1]
labels = model.classes_[::-1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_reversed, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
print("✅ Confusion Matrix:")
print(cm_reversed)

prec = cm_reversed[0,0]/ (cm_reversed[0,0] + cm_reversed[1,0])
recall = cm_reversed[0,0] / 250

print(f"Precesion: {prec}")
print(f"Precision@k: {precision_at_k}")
print(f"Recall: {recall}")
# --- 7. 特徵重要性可視化 ---
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()