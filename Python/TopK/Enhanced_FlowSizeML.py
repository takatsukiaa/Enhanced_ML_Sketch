import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
import cudf as cd
import cupy as cp
from Split_counters import split
import socket
import struct
import random

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

HOST = '0.0.0.0'
PORT = 50007
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(16)

print(f"Server listening on {HOST}:{PORT}...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

data = conn.recv(4)
(insertion_status,) = struct.unpack('i', data)

if insertion_status == 1:
    try:
        print("Insertion completed, start training model!")
        if split() != 0:
            print("Error when splitting csv file")
            exit()
    
        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_4.csv", "r") as f:
            data4 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_5.csv", "r") as f:
            data5 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_6.csv", "r") as f:
            data6 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_7.csv", "r") as f:
            data7 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

        with open("/home/takatsukiaa/ML-Sketch/Python/TopK/equinix-chicago1_output_8.csv", "r") as f:
            data8 = [list(map(int, line.strip().split(','))) for line in f if line.strip()]
        # For feature_count == 4
        columns4 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data4[0]) - 1)]
        df4 = pd.DataFrame(data4, columns=columns4)

        # For feature_count == 5
        columns5 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data5[0]) - 1)]
        df5 = pd.DataFrame(data5, columns=columns5)

        # For feature_count == 6
        columns6 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data6[0]) - 1)]
        df6 = pd.DataFrame(data6, columns=columns6)  # corrected: use data6

        # For feature_count == 7
        columns7 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data7[0]) - 1)]
        df7 = pd.DataFrame(data7, columns=columns7)

        # For feature_count == 8
        columns8 = ['feature_count', 'y'] + [f'feature_{i}' for i in range(1, len(data8[0]) - 1)]
        df8 = pd.DataFrame(data8, columns=columns8)

        models = {}  # 存各feature_count訓練好的模型
        data_by_feature = {
            4: df4,
            5: df5,
            6: df6,
            7: df7,
            8: df8
        }
        
        print("Training...")
        for feature_count, df in data_by_feature.items():
            feature_cols = [f'feature_{i}' for i in range(1, feature_count + 1)]
            df[feature_cols] = df[feature_cols].apply(lambda x: np.log1p(x))
            df['y'] = np.log1p(df['y'])
            X = df.drop(columns=['feature_count', 'y'])
            y = df['y']
            model_gb = XGBRegressor(n_estimators=55, learning_rate=0.25, max_depth=9, n_jobs=-1, device='cuda',objective='reg:absoluteerror', eval_metric='mae')
            model_gb.fit(cd.DataFrame(X),cd.DataFrame(y))
            # models[feature_count] = model_gb
            model_gb.save_model(f'/home/takatsukiaa/ML-Sketch/Sketch/TopK/model_{feature_count}.json')
        conn.sendall(struct.pack('i',1))
        conn.close()
        server_socket.close()
        print("Training is successful")
    except:
        print("Failed to Train ML Model")
        conn.sendall(struct.pack('i',-1))
        conn.close()
        server_socket.close()
        exit()

# while True:
#     # try:
#         # Receive feature vector
#         data = conn.recv(32)  # Up to 8 floats (4 bytes each)

#         if not data:
#             break

#         num_uint = len(data) // 4
#         features = struct.unpack('i' * num_uint, data)
#         if len(features) == 1:
#             (val,) = features
#             if val == -1:
#                 break
#         # print(f"Received features: {features}")
#         features = np.array(features, dtype=np.uint32)
#         features = np.reshape(features,(1,len(features)))
#         row, col = features.shape
#         # Your ML model prediction here
#         prediction = models[col].predict(features)  
#         # Send back the prediction (int)
#         prediction = int(prediction[0])
#         conn.sendall(struct.pack('i', prediction))
    # except:
        # print("Error when generating prediction!")
    # finally:
        # print("Analysis Completed")
        # conn.close()
        # server_socket.close()

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