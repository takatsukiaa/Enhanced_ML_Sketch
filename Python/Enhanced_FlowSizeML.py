import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
# df4.to_csv('convert4.csv', index=False)


# For feature_count == 5
columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data5[0]) - 1)]
df5 = pd.DataFrame(data5, columns=columns5)
# df5.to_csv('convert5.csv', index=False)


# For feature_count == 6
columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data6[0]) - 1)]
df6 = pd.DataFrame(data6, columns=columns6)  # corrected: use data6
# df6.to_csv('convert6.csv', index=False)

# For feature_count == 7
columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data7[0]) - 1)]
df7 = pd.DataFrame(data7, columns=columns7)
# df7.to_csv('convert7.csv', index=False)

# For feature_count == 8
columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data8[0]) - 1)]
df8 = pd.DataFrame(data7, columns=columns7)
# # df8.to_csv('convert8.csv', index=False)

data_by_feature = {
    4: df4,
    5: df5,
    6: df6,
    7: df7,
    8: df8
}

degree = 2
models = {}   # Dictionary to hold models, keyed by feature count
results = {}  # Dictionary to hold evaluation results

models_poly = {}
results_poly = {}

models_tree = {}
results_tree = {}

for feature_count, df in data_by_feature.items():
    # Assume df has columns: ['feature_count', 'y', 'feature_1', 'feature_2', ...]
    # Use all columns except for 'feature_count' and 'y' as predictors.
    X = df.drop(columns=['feature_count', 'y'])
    y = df['y']
    
    # 分割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Gradient Boosting model
    # model_gb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7)
    # model_gb.fit(X_train,y_train)
    # y_pred_gb = model_gb.predict(X_test)
    # mse_gb = mean_squared_error(y_test,y_pred_gb)
    # r2_tree = r2_score(y_test,y_pred_gb)
    # print("Gradient Boosting:")
    # print(f"Feature Count {feature_count}")
    # print(f"Mean Squared Error: {mse_gb:.4f}")
    # print(f"R² Score: {r2_tree:.4f}")

    # Train random forest model
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=9, random_state=42)
    model_rf.fit(X_train,y_train)
    y_pred_tree = model_rf.predict(X_test)
    mse_tree = mean_squared_error(y_test,y_pred_tree)
    r2_tree = r2_score(y_test,y_pred_tree)
    print("Random Forest:")
    print(f"Feature Count {feature_count}")
    print(f"Mean Squared Error: {mse_tree:.4f}")
    print(f"R² Score: {r2_tree:.4f}")
    
    # Train decision tree model
    # model_tree = DecisionTreeRegressor(max_depth=7)
    # model_tree.fit(X_train,y_train)
    # y_pred_tree = model_tree.predict(X_test)
    # mse_tree = mean_squared_error(y_test,y_pred_tree)
    # r2_tree = r2_score(y_test,y_pred_tree)
    # print("Decision Tree:")
    # print(f"Feature Count {feature_count}")
    # print(f"Mean Squared Error: {mse_tree:.4f}")
    # print(f"R² Score: {r2_tree:.4f}")
    

    
    # Train the polynomial model
    # model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # model_poly.fit(X_train, y_train)
    # y_pred_poly = model_poly.predict(X_test)
    # mse_poly = mean_squared_error(y_test,y_pred_poly)
    # r2_poly = r2_score(y_test,y_pred_poly)
    
    # print("Poly:")
    # print(f"Feature Count {feature_count}")
    # print(f"Mean Squared Error: {mse_poly:.4f}")
    # print(f"R² Score: {r2_poly:.4f}")
    
    # Scale the data
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # Train the Ridge Model
    # model = Ridge(alpha=1.0)  # Set regularization strength (α)
    # models[feature_count] = model
    # model.fit(X_train_scaled, y_train)

    # y_pred_ridge = model.predict(X_test_scaled)

    # mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    # r2_ridge = r2_score(y_test, y_pred_ridge)

    # print("Ridge:")
    # print(f"Feature Count {feature_count}")
    # print(f"Mean Squared Error: {mse_ridge:.4f}")
    # print(f"R² Score: {r2_ridge:.4f}")


