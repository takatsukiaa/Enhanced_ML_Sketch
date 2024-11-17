import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np


def average_relative_error(y_true, y_pred):
    """
    計算 Average Relative Error (ARE)
    
    Parameters:
    - y_true: 真實值 (numpy array or list)
    - y_pred: 預測值 (numpy array or list)
    
    Returns:
    - ARE: 平均相對誤差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 防止除以 0，將 0 的地方加上小平滑值
    epsilon = 1e-10  # 可調整的小值
    relative_errors = np.abs((y_true - y_pred) / (y_true + epsilon))
    
    return np.mean(relative_errors)


## usage: python3 FlowSizeML.py <input_file> <output_file>
## input_file: 輸入資料檔案(actual flow size , counter1, counter2, counter3)
## output_file: 輸出模型參數檔案(mean1, mean2, mean3, 0, scale1, scale2, scale3, 1, coef1, coef2, coef3, 0), each counter has a mean, a scale and a coef

##mian
SCALING_RATE = 1000  # 將流量大小縮小的比例
print("FlowSizeML.py: Start to run...")

# 讀取資料
train_mat = np.loadtxt(open(sys.argv[1], "rb"), delimiter=" ", skiprows=0)
f = open(sys.argv[2], "w")  # 開啟輸出檔案
print("data size: ", train_mat.shape)
THRESH = (int)(train_mat.shape[0] / SCALING_RATE)  # set threshold for filtering mice flows
print("Threshold: ", THRESH)

# 使用 20% 的資料進行預測
predict_idx = np.random.choice(train_mat.shape[0], int(train_mat.shape[0] * 0.2), replace=False)
remaining_idx = np.setdiff1d(np.arange(train_mat.shape[0]), predict_idx)
predict_X = train_mat[predict_idx, 1:4].copy()
predict_Y = train_mat[predict_idx, 0].copy()

# 剩下的 80% 資料分為訓練集和測試集
train_idx = np.random.choice(remaining_idx, int(len(remaining_idx) * 0.8), replace=False)
test_idx = np.setdiff1d(remaining_idx, train_idx)
train_X = train_mat[train_idx, 1:4].copy()
train_Y = train_mat[train_idx, 0].copy()
test_X = train_mat[test_idx, 1:4].copy()
test_Y = train_mat[test_idx, 0].copy()

# 篩選小於 THRESH 的資料
mask_train = train_Y < THRESH
train_X = train_X[mask_train]
train_Y = train_Y[mask_train]

mask_test = test_Y < THRESH
test_X = test_X[mask_test]
test_Y = test_Y[mask_test]

mask_predict = predict_Y < THRESH
predict_X = predict_X[mask_predict]
predict_Y = predict_Y[mask_predict]

# 標準化處理
feature_scaler = preprocessing.StandardScaler().fit(train_X)
train_X = feature_scaler.transform(train_X)

# 訓練模型
reg = linear_model.Ridge(alpha=0.5)
reg.fit(train_X, train_Y)

# 測試資料預測及誤差計算
test_X = feature_scaler.transform(test_X)
predicted_test_Y = reg.predict(test_X)
mae_test = mean_absolute_error(test_Y, predicted_test_Y)
are_test = average_relative_error(test_Y, predicted_test_Y)
print(f"Test MAE: {mae_test:.2f}")
print(f"Test ARE: {are_test:.2f}")

# 20% 資料預測及誤差計算
predict_X = feature_scaler.transform(predict_X)
predicted_Y = reg.predict(predict_X)
mae_predict = mean_absolute_error(predict_Y, predicted_Y)
are_predict = average_relative_error(predict_Y, predicted_Y)
print(f"Predict MAE (20% Data): {mae_predict:.2f}")
print(f"Predict ARE (20% Data): {are_predict:.2f}")

# 將模型參數輸出到檔案
for i in range(train_X.shape[1]):
    print(feature_scaler.mean_[i], file=f)
print("0", file=f)  # 占位符
for i in range(train_X.shape[1]):
    print(feature_scaler.scale_[i], file=f)
print("1", file=f)  # 占位符
for i in range(train_X.shape[1]):
    print(reg.coef_[i], file=f)
print("0", file=f)  # 占位符

print("FlowSizeML.py: Finish running.")