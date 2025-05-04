import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
import cudf as cd
import cupy as cp

# 自訂 MRE（Mean Relative Error）
def mean_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    relative_error = np.abs(y_true - y_pred) / np.abs(y_true)
    relative_error = relative_error[~np.isinf(relative_error)]  # 避免除零錯誤
    return np.mean(relative_error)

# 自訂 MAE/MRE on 原始scale
def mae_on_original_scale(y_true_log, y_pred_log):
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    return mean_absolute_error(y_true_original, y_pred_original)

def mre_on_original_scale(y_true_log, y_pred_log):
    y_true_original = np.expm1(y_true_log)
    y_pred_original = np.expm1(y_pred_log)
    return mean_relative_error(y_true_original, y_pred_original)

def is_model_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except:
        return False

# 讀入資料
train_files = {
    4: "equinix-chicago1_output_4_1.csv",
    5: "equinix-chicago1_output_5_1.csv",
    6: "equinix-chicago1_output_6_1.csv",
    7: "equinix-chicago1_output_7_1.csv",
    8: "equinix-chicago1_output_8_1.csv",
}

# test_files = {
#     4: "equinix-chicago1_output_4_2.csv",
#     5: "equinix-chicago1_output_5_2.csv",
#     6: "equinix-chicago1_output_6_2.csv",
#     7: "equinix-chicago1_output_7_2.csv",
#     8: "equinix-chicago1_output_8_2.csv",
# }

# 用pandas讀
data_by_feature = {k: pd.read_csv(v) for k, v in train_files.items()}
# test_data_by_feature = {k: pd.read_csv(v) for k, v in test_files.items()}

models = {}  # 存各feature_count訓練好的模型
print("Training...")

# 訓練階段
total_rows, total_mae, total_mre, total_mae_original, total_mre_original = 0, 0, 0, 0, 0
# 準備好新的預測結果檔案
train_result_csv = open("training_prediction_result.csv", "w", newline='')
train_writer = csv.writer(train_result_csv)
train_writer.writerow(["ID", "topk_flag", "predicted_size", "true_size"])


for feature_count, df in data_by_feature.items():
    feature_cols = [f"{i}" for i in range(1, feature_count + 1)]
    X = df[feature_cols]
    y = df["actual_size"]

    ids = df["ID"]
    flags = df["topk_flag"]

    X = X.apply(np.log1p)
    y = np.log1p(y)

    model = XGBRegressor(n_estimators=55, learning_rate=0.25, max_depth=9, n_jobs=-1, device="cuda", objective='reg:absoluteerror', eval_metric='mae')
    model.fit(cd.DataFrame(X), cd.DataFrame(y))
    models[feature_count] = model

    y_pred_log = model.predict(X)

    # 還原
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y)

    # 存每一筆
    for i in range(len(ids)):
        train_writer.writerow([ids.iloc[i], flags.iloc[i], y_pred[i], y_true.iloc[i]])

# 關閉訓練結果檔案
train_result_csv.close()
# ------------------------------------testing------------------------------------
# print("\nTesting...")
# # 測試結果檔案
# test_result_csv = open("testing_prediction_result.csv", "w", newline='')
# test_writer = csv.writer(test_result_csv)
# test_writer.writerow(["ID", "topk_flag", "predicted_size", "true_size"])

# # 測試階段
# total_rows, total_mae, total_mre, total_mae_original, total_mre_original = 0, 0, 0, 0, 0

# for feature_count, test_df in test_data_by_feature.items():
#     feature_cols = [f"{i}" for i in range(1, feature_count + 1)]

#     X = test_df[feature_cols]
#     y = test_df["actual_size"]

#     X = X.apply(np.log1p)
#     y = np.log1p(y)

#     model = models[feature_count]

#     if is_model_fitted(model):
#         y_pred = model.predict(X)
#     else:
#         print(f"Model for feature_count {feature_count} not fitted!")
#         continue

#     y_pred_original = np.expm1(y_pred)
#     y_original = np.expm1(y)
#     # 寫入.csv紀錄
#     for i in range(len(test_df)):
#         test_writer.writerow([test_df["ID"].iloc[i], test_df["topk_flag"].iloc[i], y_pred_original[i], y_original.iloc[i]])
    
#     # 評估

#     mae = mean_absolute_error(y, y_pred)
#     mre = mean_relative_error(y, y_pred)
#     mae_original = mean_absolute_error(y_original, y_pred_original)
#     mre_original = mean_relative_error(y_original, y_pred_original)

#     print(f"Feature Count {feature_count} - Test:")
#     print(f"  MAE: {mae:.4f}, MRE: {mre:.4f}")
#     print(f"  MAE (Original Scale): {mae_original:.4f}, MRE (Original Scale): {mre_original:.4f}")

#     total_rows += len(test_df)
#     total_mae += mae * len(test_df)
#     total_mre += mre * len(test_df)
#     total_mae_original += mae_original * len(test_df)
#     total_mre_original += mre_original * len(test_df)

# print("\nTesting Overall:")
# print(f"MAE: {total_mae/total_rows:.4f}")
# print(f"MRE: {total_mre/total_rows:.4f}")
# print(f"MAE (Original Scale): {total_mae_original/total_rows:.4f}")
# print(f"MRE (Original Scale): {total_mre_original/total_rows:.4f}")


# ##TopK: build heap for 