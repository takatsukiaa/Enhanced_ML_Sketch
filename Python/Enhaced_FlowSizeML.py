import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
# 讀取檔案

with open("/home/takatsukiaa/ML-Sketch/Python/output_4.csv", "r") as f:
    data4 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/output_5.csv", "r") as f:
    data5 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/output_6.csv", "r") as f:
    data6 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/output_7.csv", "r") as f:
    data7 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/output_8.csv", "r") as f:
    data8 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# 轉換為 DataFrame


# For feature_count == 4
columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data4[0]) - 1)]
df4 = pd.DataFrame(data4, columns=columns4)
df4.to_csv('convert4.csv', index=False)


# For feature_count == 5
columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data5[0]) - 1)]
df5 = pd.DataFrame(data5, columns=columns5)
df5.to_csv('convert5.csv', index=False)


# For feature_count == 6
columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data6[0]) - 1)]
df6 = pd.DataFrame(data6, columns=columns6)  # corrected: use data6
df6.to_csv('convert6.csv', index=False)

# For feature_count == 7
columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data7[0]) - 1)]
df7 = pd.DataFrame(data7, columns=columns7)
df7.to_csv('convert7.csv', index=False)

# For feature_count == 8
columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data8[0]) - 1)]
df8 = pd.DataFrame(data7, columns=columns7)
df8.to_csv('convert8.csv', index=False)

data_by_feature = {
    4: df4,
    5: df5,
    6: df6,
    7: df7,
    8: df8
}

models = {}   # Dictionary to hold models, keyed by feature count
results = {}  # Dictionary to hold evaluation results

for feature_count, df in data_by_feature.items():
    # Assume df has columns: ['feature_count', 'y', 'feature_1', 'feature_2', ...]
    # Use all columns except for 'feature_count' and 'y' as predictors.
    X = df.drop(columns=['feature_count', 'y'])
    y = df['y']
    
    # 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression().fit(X_train, y_train)
    models[feature_count] = model
    
    # Get predictions and compute evaluation metrics (e.g., MSE and R^2)
    y_pred = model.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X, y)
    
    # Store the results in a nested dictionary
    results[feature_count] = {
        "predictions": y_pred,
        "mse": mse,
        "r2": r2
    }
# Now you can access the model and results for each feature count easily:
print("Models:", models)
print("Results:", results)

# 分組處理不同feature count的資料
# for feature_count in [3, 4, 5, 6]:
#     # 過濾出符合條件的資料
#     filtered_data = df[df['feature_count'] == feature_count]
#     # print(filtered_data.head())
#     # 提取特徵和標籤
#     X = filtered_data.iloc[:, 2:feature_count + 2].values
#     y = filtered_data['y'].values  # 1 th是 target value
#     # print(f"Features (X):\n{X[:5]}")
#     # print(f"Labels (y):\n{y[:5]}")
#     if len(X) == 0:
#         print(f"No data for feature count = {feature_count}")
#         continue

#     # 分割訓練集與測試集
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # 訓練線性迴歸模型
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # 預測與評估
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
    
#     # 儲存模型與結果
#     models[feature_count] = model
#     results[feature_count] = {"MSE": mse, "Model": model}
    
#     print(f"Feature count {feature_count}: MSE = {mse:.4f}")
