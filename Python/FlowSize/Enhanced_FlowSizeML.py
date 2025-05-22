import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest, mutual_info_regression
# from sklearn.linear_model import HuberRegressor
import cudf as cd
import cupy as cp


with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_4.csv", "r") as f:
    data4 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_5.csv", "r") as f:
    data5 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_6.csv", "r") as f:
    data6 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_7.csv", "r") as f:
    data7 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_8.csv", "r") as f:
    data8 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_4_2.csv", "r") as f:
    test_data4 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_5_2.csv", "r") as f:
    test_data5 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_6_2.csv", "r") as f:
    test_data6 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_7_2.csv", "r") as f:
    test_data7 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/FlowSize/equinix-chicago1_output_8_2.csv", "r") as f:
    test_data8 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

# 轉換為 DataFrame


# For feature_count == 4
columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data4[0]) - 1)]
df4 = pd.DataFrame(data4, columns=columns4)
test_columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data4[0]) - 1)]
test_df4 = pd.DataFrame(test_data4, columns=test_columns4)

# For feature_count == 5
columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data5[0]) - 1)]
df5 = pd.DataFrame(data5, columns=columns5)
test_columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data5[0]) - 1)]
test_df5 =  pd.DataFrame(test_data5, columns=test_columns5)

# For feature_count == 6
columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data6[0]) - 1)]
df6 = pd.DataFrame(data6, columns=columns6)  # corrected: use data6
test_columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data6[0]) - 1)]
test_df6 = pd.DataFrame(test_data6, columns=test_columns6)

# For feature_count == 7
columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data7[0]) - 1)]
df7 = pd.DataFrame(data7, columns=columns7)
test_columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data7[0]) - 1)]
test_df7 = pd.DataFrame(test_data7, columns=test_columns7)

# For feature_count == 8
columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data8[0]) - 1)]
df8 = pd.DataFrame(data8, columns=columns8)
test_columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(test_data8[0]) - 1)]
test_df8 = pd.DataFrame(test_data8, columns=test_columns8)

data_by_feature = {
    4: df4,
    5: df5,
    6: df6,
    7: df7,
    8: df8
}

test_data_by_feature = {
    4: test_df4,
    5: test_df5,
    6: test_df6,
    7: test_df7,
    8: test_df8
}

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
total_mre_original = 0
total_mae_original = 0
degree = 2
models = {}   # Dictionary to hold models, keyed by feature count
results = {}  # Dictionary to hold evaluation results

print("Gradient Boosting:")

def mae_on_original_scale(y_true_log, y_pred_log):
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    return mean_absolute_error(y_true_original, y_pred_original)

def mre_on_original_scale(y_true_log, y_pred_log):
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    return mean_relative_error(y_true_original, y_pred_original)

for feature_count, df in data_by_feature.items():
    # Assume df has columns: ['feature_count', 'y', 'feature_1', 'feature_2', ...]
    # Use all columns except for 'feature_count' and 'y' as predictors.
    model_gb = XGBRegressor(n_estimators=55, learning_rate=0.25, max_depth=9, n_jobs=-1, device="cuda", objective='reg:absoluteerror', eval_metric='mae')
    model_rf = RandomForestRegressor(n_estimators=55, max_depth=9, random_state=42, n_jobs=-1)
    
    # 分割訓練集與測試集
    feature_cols = [f'feature_{i}' for i in range(1, feature_count + 1)]
    df[feature_cols] = df[feature_cols].apply(lambda x: np.log1p(x))
    df['y'] = np.log1p(df['y'])
    X = df.drop(columns=['feature_count', 'y'])
    y = df['y']
    if len(df.index) > 5000:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_gb.fit(cd.DataFrame(X_train),cd.DataFrame(y_train))
        models[feature_count] = model_gb
        y_pred_gb = model_gb.predict(cd.DataFrame(X_test))
        y_pred_original = np.expm1(y_pred_gb)
        y_test_original = np.expm1(y_test)
        mae = mean_absolute_error(y_test,y_pred_gb)
        mre = mean_relative_error(y_test,y_pred_gb)
        mae_original = mean_absolute_error(y_test_original,y_pred_original)
        mre_original = mean_relative_error(y_test_original,y_pred_original)
        print(f"Feature Count {feature_count}")
        print(f"Training Mean Absolute Error: {mae:.4f}" )
        print(f"Training Mean Relative Error:{mre:.4f}")
        print(f"Training Mean Absolute Error Transformed Back: {mae_original:.4f}" )
        print(f"Training Mean Relative Error Transformed Back: {mre_original:.4f}")        
        # else:
        # # Train random forest model
        #     model_rf.fit(X_train,y_train)       
        #     models[feature_count] = model_rf
        #     y_pred_rf = model_rf.predict(X_test)
        #     y_pred_original = np.expm1(y_pred_rf)
        #     y_test_original = np.expm1(y_test)
        #     mae_original = mean_absolute_error(y_test_original,y_pred_original)
        #     mre_original = mean_relative_error(y_test_original,y_pred_original)
        #     mre = mean_relative_error(y_test,y_pred_rf)
        #     mae = mean_absolute_error(y_test, y_pred_rf)
        #     print(f"Feature Count {feature_count}")
        #     print(f"Mean Absolute Error: {mae:.4f}" )            y_pred_original = np.expm1(y_pred_gb)
        y_test_original = np.expm1(y_test)
        mae = mean_absolute_error(y_test,y_pred_gb)
        mre = mean_relative_error(y_test,y_pred_gb)
        mae_original = mean_absolute_error(y_test_original,y_pred_original)
        mre_original = mean_relative_error(y_test_original,y_pred_original)
        # print(f"Mean Absolute Error Transformed Back: {mae_original:.4f}" )
        # print(f"Mean Relative Error Transformed Back: {mre_original:.4f}")
    else:
        # Train Gradient Boosting model
        y_pred_gb = cross_val_predict(model_gb, X, y, cv=5)
        y_pred_original = np.expm1(y_pred_gb)
        y_test_original = np.expm1(y)
        mre_original_scorer = make_scorer(mre_on_original_scale, greater_is_better=False)
        mae_original_scorer = make_scorer(mae_on_original_scale, greater_is_better=False)
        mre_scorer = make_scorer(mean_relative_error, greater_is_better=False)
        mre = cross_val_score(model_gb, X, y, scoring=mre_scorer, cv=5)
        mre = -mre
        mae = cross_val_score(model_gb, X, y, cv=5, scoring="neg_mean_absolute_error")
        mae = -mae
        mre_original = cross_val_score(model_gb, X, y, cv=5, scoring=mre_original_scorer)
        mre_original = -mre_original
        mae_original = cross_val_score(model_gb, X, y, cv=5, scoring=mae_original_scorer)
        mae_original = -mae_original
        # Train random forest model
        # mre_scorer = make_scorer(mean_relative_error, greater_is_better=False)
        # mre = cross_val_score(model_rf, X, y, scoring=mre_scorer, cv=5)
        # mre = -mre
        # mae = cross_val_score(model_rf, X, y, cv=5, scoring="neg_mean_absolute_error")
        # mae = -mae
        print(f"Feature Count {feature_count}")
        print(f"Mean Absolute Error: {np.mean(mae):.4f}")
        print(f"Mean Relative Error:{np.mean(mre):.4f}")
        print(f"Mean Absolute Error Transformed Back: {np.mean(mae_original):.4f}" )
        print(f"Mean Relative Error Transformed Back: {np.mean(mre_original):.4f}")

    total_rows += len(df.index)
    total_mre += np.mean(mre) * len(df.index)
    total_mae += np.mean(mae) * len(df.index)
    total_mre_original += np.mean(mre_original) * len(df.index)
    total_mae_original += np.mean(mae_original) * len(df.index)

mean_mae = total_mae / total_rows
mean_mre = total_mre / total_rows
mean_mae_original = total_mae_original / total_rows
mean_mre_original = total_mre_original / total_rows
print(f"Training Overall MAE:{mean_mae:.4f}")
print(f"Training Overall MRE:{mean_mre:.4f}")
print(f"Training Overall MAE Transformed Back:{mean_mae_original:.4f}")
print(f"Training Overall MRE Transformed Back:{mean_mre_original:.4f}")

mean_mae_original = 0
mean_mae_original = 0
mean_mae = 0
mean_mre = 0
total_mae = 0
total_mre = 0
total_rows = 0
total_mre_original = 0
total_mae_original = 0

print("\nTesting with second half")

def is_model_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except:
        return False
    
for feature_count, test_df in test_data_by_feature.items(): 
    feature_cols = [f'feature_{i}' for i in range(1, feature_count + 1)]
    test_df[feature_cols] = test_df[feature_cols].apply(lambda x: np.log1p(x))
    test_df['y'] = np.log1p(test_df['y'])
    X = test_df.drop(columns=['feature_count', 'y'])
    y = test_df['y']
    model = models[feature_count]
    if len(test_df.index) > 5000 and (is_model_fitted(model)):
        y_pred_gb = model.predict(cd.DataFrame(X))
        mre = mean_relative_error(y,y_pred_gb)
        mae = mean_absolute_error(y, y_pred_gb)
        y_pred_original = np.expm1(y_pred_gb)
        y_original = np.expm1(y)
        mae = mean_absolute_error(y,y_pred_gb)
        mre = mean_relative_error(y,y_pred_gb)
        mae_original = mean_absolute_error(y_original,y_pred_original)
        mre_original = mean_relative_error(y_original,y_pred_original)
        print(f"Feature Count {feature_count}")
        print(f"Mean Absolute Error: {mae:.4f}" )
        print(f"Mean Relative Error : {mre:.4f}")
        print(f"Training Mean Absolute Error Transformed Back: {mae_original:.4f}" )
        print(f"Training Mean Relative Error Transformed Back: {mre_original:.4f}")
    else:
        mre_scorer = make_scorer(mean_relative_error, greater_is_better=False)
        mre = cross_val_score(model, X, y, cv=5, scoring=mre_scorer)
        mre = -mre
        mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        mae = -mae
        print(f"Feature Count {feature_count}")
        print(f"Mean Absolute Error: {np.mean(mae):.4f}")
        print(f"Mean Relative Error: {np.mean(mre):.4f}")
    total_rows += len(test_df.index)
    total_mre += np.mean(mre) * len(test_df.index)
    total_mae += np.mean(mae) * len(test_df.index)
    total_mre_original += np.mean(mre_original) * len(test_df.index)
    total_mae_original += np.mean(mae_original) * len(test_df.index)
mean_mae = total_mae / total_rows
mean_mre = total_mre / total_rows
mean_mae_original = total_mae_original / total_rows
mean_mre_original = total_mre_original / total_rows
print(f"Testing Overall MAE:{mean_mae:.4f}")
print(f"Testing Overall MRE:{mean_mre:.4f}")
print(f"Testing Overall MAE Transformed Back:{mean_mae_original:.4f}")
print(f"Testing Overall MRE Transformed Back:{mean_mre_original:.4f}")

