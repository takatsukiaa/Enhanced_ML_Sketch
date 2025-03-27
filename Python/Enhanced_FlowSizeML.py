import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_4.csv", "r") as f:
    data4 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_5.csv", "r") as f:
    data5 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_6.csv", "r") as f:
    data6 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_7.csv", "r") as f:
    data7 = [list(map(int, line.strip().split())) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_8.csv", "r") as f:
    data8 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_4_2.csv", "r") as f:
#     test_data4 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_5_2.csv", "r") as f:
#     test_data5 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_6_2.csv", "r") as f:
#     test_data6 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_7_2.csv", "r") as f:
#     test_data7 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# with open("/home/takatsukiaa/ML-Sketch/Python/equinix-chicago1_output_8_2.csv", "r") as f:
#     test_data8 = [list(map(int, line.strip().split())) for line in f if line.strip()]

# 轉換為 DataFrame


# For feature_count == 4
columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data4[0]) - 1)]
df4 = pd.DataFrame(data4, columns=columns4)
# df4.to_csv('convert4.csv', index=False)
# test_columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data4[0]) - 1)]
# test_df4 = pd.DataFrame(test_data4, columns=test_columns4)

# For feature_count == 5
columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data5[0]) - 1)]
df5 = pd.DataFrame(data5, columns=columns5)
# df5.to_csv('convert5.csv', index=False)
# test_columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data5[0]) - 1)]
# test_df5 =  pd.DataFrame(test_data5, columns=test_columns5)

# For feature_count == 6
columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data6[0]) - 1)]
df6 = pd.DataFrame(data6, columns=columns6)  # corrected: use data6
# df6.to_csv('convert6.csv', index=False)
# test_columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data6[0]) - 1)]
# test_df6 = pd.DataFrame(test_data6, columns=test_columns6)

# For feature_count == 7
columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data7[0]) - 1)]
df7 = pd.DataFrame(data7, columns=columns7)
# df7.to_csv('convert7.csv', index=False)
# test_columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data7[0]) - 1)]
# test_df7 = pd.DataFrame(test_data7, columns=test_columns7)

# For feature_count == 8
columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data8[0]) - 1)]
df8 = pd.DataFrame(data8, columns=columns8)
# df8.to_csv('convert8.csv', index=False)
# test_columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data8[0]) - 1)]
# test_df8 = pd.DataFrame(test_data8, columns=test_columns8)

data_by_feature = {
    4: df4,
    5: df5,
    6: df6,
    7: df7,
    8: df8
}

# test_data_by_feature = {
#     4: test_df4,
#     5: test_df5,
#     6: test_df6,
#     7: test_df7,
#     8: test_df8
# }

def mean_relative_error(y_true, y_pred):
    """
    Calculate the Mean Relative Error (MRE) between true and predicted values.

    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values

    Returns:
    float: Mean Relative Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    relative_error = np.abs(y_true - y_pred) / np.abs(y_true)
    
    # Avoid division by zero
    relative_error = relative_error[~np.isinf(relative_error)]

    return np.mean(relative_error)

total_rows = 0
mean_mae = 0
mean_mre = 0
total_mre = 0
total_mae = 0
degree = 2
models = {}   # Dictionary to hold models, keyed by feature count
results = {}  # Dictionary to hold evaluation results

# print("Gradient Boosting:")
print("Random Forest:")
for feature_count, df in data_by_feature.items():
    # Assume df has columns: ['feature_count', 'y', 'feature_1', 'feature_2', ...]
    # Use all columns except for 'feature_count' and 'y' as predictors.
    X = df.drop(columns=['feature_count', 'y'])
    y = df['y']
    # model_gb = XGBRegressor(n_estimators=55, learning_rate=0.1, max_depth=9)
    model_rf = RandomForestRegressor(n_estimators=55, max_depth=10, random_state=42, n_jobs=-1, criterion='absolute_error')
    # 分割訓練集與測試集
    if len(df.index) > 5000:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train Gradient Boosting model
        # model_gb.fit(X_train,y_train)
        # y_pred_gb = model_gb.predict(X_test)
        # mae = mean_absolute_error(y_test,y_pred_gb)
        # mre = mean_relative_error(y_test,y_pred_gb)
        # Train random forest model
        model_rf.fit(X_train,y_train)
        models[feature_count] = model_rf
        y_pred_rf = model_rf.predict(X_test)
        # y_pred_all = model_rf.predict(X)
        mre = mean_relative_error(y_test,y_pred_rf)
        mae = mean_absolute_error(y_test, y_pred_rf)
        print(f"Feature Count {feature_count}")
        print(f"Mean Absolute Error: {mae:.4f}" )
        print(f"Mean Relative Error:{mre:.4f}")
        # comparison_df = pd.DataFrame({
        #     'Actual': y.values,
        #     'Predicted': y_pred_all
        # })
        # comparison_df.to_csv(f'comparison_feature_{feature_count}_full.csv', index=False)
    else:
        # Train Gradient Boosting model
        # mre_scorer = make_scorer(mean_relative_error, greater_is_better=False)
        # mre = cross_val_score(model_gb, X, y, scoring=mre_scorer, cv=5)
        # mre = -mre
        # mae = cross_val_score(model_gb, X, y, cv=5, scoring="neg_mean_absolute_error")
        # mae = -mae
        
        # Train random forest model
        mre_scorer = make_scorer(mean_relative_error, greater_is_better=False)
        mre = cross_val_score(model_rf, X, y, scoring=mre_scorer, cv=5)
        mre = -mre
        mae = cross_val_score(model_rf, X, y, cv=5, scoring="neg_mean_absolute_error")
        mae = -mae
        print(f"Feature Count {feature_count}")
        print(f"Mean Absolute Error: {np.mean(mae):.4f}")
        print(f"Mean Relative Error:{np.mean(mre):.4f}")

    total_rows += len(df.index)
    total_mre += np.mean(mre) * len(df.index)
    total_mae += np.mean(mae) * len(df.index)


    
    # Train decision tree model
    # if len(df.index) > 1000:
    #     model_tree = DecisionTreeRegressor(max_depth=9)
    #     model_tree.fit(X_train,y_train)
    #     y_pred_tree = model_tree.predict(X_test)
    #     mae_tree = mean_absolute_error(y_test, y_pred_tree)
    #     r2_tree = r2_score(y_test,y_pred_tree)
    #     print("Decision Tree:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Absolute Error: {mae_tree:.4f}")
    #     print(f"R² Score: {r2_tree:.4f}")
    # else:
    #     model_tree = DecisionTreeRegressor(max_depth=9)
    #     r2_tree = cross_val_score(model_tree, X, y, cv=5, scoring="r2")
    #     mae_tree = cross_val_score(model_tree, X, y, cv=5, scoring="neg_mean_absolute_error")
    #     mae_tree = -mae_tree
    #     print("Decision Tree:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Absolute Error: {np.mean(mae_tree):.4f}")
    #     print(f"R² Score: {r2_tree.mean():.4f}")
    

    
    # Train the polynomial model
    # model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # if len(df.index) > 1000:
    #     model_poly.fit(X_train, y_train)
    #     y_pred_poly = model_poly.predict(X_test)
    #     mae_poly = mean_absolute_error(y_test,y_pred_poly)
    #     r2_poly = r2_score(y_test,y_pred_poly)
    #     print("Poly:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Absolute Error: {mae_poly:.4f}")
    #     print(f"R² Score: {r2_poly:.4f}")
    # else:
    #     r2_poly = cross_val_score(model_poly, X, y, cv=5, scoring="r2")
    #     mae_poly = cross_val_score(model_poly, X, y, cv=5, scoring="neg_mean_absolute_error")
    #     mae_poly = -mae_poly
    #     print("Poly:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Absolute Error: {np.mean(mae_poly):.4f}")
    #     print(f"R² Score: {r2_poly.mean():.4f}")

    
    # Scale the data
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Train the Ridge Model
    # model = Ridge(alpha=1)  # Set regularization strength (α)
    # models[feature_count] = model
    # model.fit(X_train_scaled, y_train)

    # y_pred_ridge = model.predict(X_test_scaled)

    # mse_ridge = mean_absolute_error(y_test, y_pred_ridge)
    # r2_ridge = r2_score(y_test, y_pred_ridge)

    # print("Ridge:")
    # print(f"Feature Count {feature_count}")
    # print(f"Mean Squared Error: {mse_ridge:.4f}")
    # print(f"R² Score: {r2_ridge:.4f}")

    # Training the Linear Regression model
    # model_linear = LinearRegression()
    # if len(df.index) > 1000:
    #     model_linear.fit(X_train,y_train)
    #     y_pred_linear = model_linear.predict(X_test)
    #     mae_linear = mean_absolute_error(y_test,y_pred_linear)
    #     r2_linear = r2_score(y_test,y_pred_linear)
    #     print("Linear Regression:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Absolute Error: {mae_linear:.4f}")
    #     print(f"R² Score: {r2_linear:.4f}")
    # else:
    #     r2_linear = cross_val_score(model_linear, X, y, cv=5, scoring="r2")
    #     mae_linear = cross_val_score(model_linear, X, y, cv=5, scoring="neg_mean_absolute_error")
    #     mae_linear = -mae_linear
    #     print("Linear Regression:")
    #     print(f"Feature Count {feature_count}")
    #     print(f"Mean Squared Error: {np.mean(mae_linear):.4f}")
    #     print(f"R² Score: {r2_linear.mean():.4f}")

mean_mae = total_mae / total_rows
mean_mre = total_mre / total_rows
print(f"Overall MAE:{mean_mae:.4f}")
print(f"Overall MRE:{mean_mre:.4f}")

mean_mae = 0
mean_mre = 0
total_mae = 0
total_mre = 0
total_rows = 0
# print("Testing with equinix-chicago2")

# def is_model_fitted(model):
#     try:
#         check_is_fitted(model)
#         return True
#     except:
#         return False
    
# for feature_count, test_df in test_data_by_feature.items(): 
#     X = test_df.drop(columns=['feature_count', 'y'])
#     y = test_df['y']
#     model = models[feature_count]
#     if len(test_df.index) > 1000 and (is_model_fitted(model)):
#         y_pred_rf = model.predict(X)
#         r2_rf = r2_score(y,y_pred_rf)
#         mae_rf = mean_absolute_error(y, y_pred_rf)
#         print(f"Feature Count {feature_count}")
#         print(f"Mean Absolute Error: {mae_rf:.4f}" )
#         print(f"R² Score: {r2_rf:.4f}")
#     else:
#         r2_rf = cross_val_score(model, X, y, cv=5, scoring="r2") # 5-Fold Cross-Validation
#         mae_rf = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
#         mae_rf = -mae_rf
#         print(f"Feature Count {feature_count}")
#         print(f"Mean Absolute Error: {np.mean(mae_rf):.4f}")
#         print(f"R² Score: {r2_rf.mean():.4f}")

