import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
# 讀取檔案

file_path = os.path.join(os.getcwd(), "counters.txt")
data = []

with open(file_path, "r") as f:
    for line in f:
        values = list(map(int, line.strip().split()))
        data.append(values)

# 轉換為 DataFrame
columns = ['feature_count'] +['y'] +[f'feature_{i}' for i in range(1, len(data[0])-1)]
df = pd.DataFrame(data, columns=columns)
df.to_csv('convert.csv', index=False)
models = {}
results = {}

# 分組處理不同feature count的資料
for feature_count in [3, 4, 5, 6]:
    # 過濾出符合條件的資料
    filtered_data = df[df['feature_count'] == feature_count]
    # print(filtered_data.head())
    # 提取特徵和標籤
    X = filtered_data.iloc[:, 2:feature_count + 2].values
    y = filtered_data['y'].values  # 1 th是 target value
    # print(f"Features (X):\n{X[:5]}")
    # print(f"Labels (y):\n{y[:5]}")
    if len(X) == 0:
        print(f"No data for feature count = {feature_count}")
        continue

    # 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 訓練線性迴歸模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 預測與評估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # 儲存模型與結果
    models[feature_count] = model
    results[feature_count] = {"MSE": mse, "Model": model}
    
    print(f"Feature count {feature_count}: MSE = {mse:.4f}")
