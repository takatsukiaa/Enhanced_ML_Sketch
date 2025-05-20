import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import cudf as cd


with open("/home/takatsukiaa/ML-Sketch/Python/TopK/training_flows.csv", "r") as f:
    training_data = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

with open("/home/takatsukiaa/ML-Sketch/Python/TopK/testing_flows.csv", "r") as f:
    testing_data = [list(map(int, line.strip().split(','))) for line in f if line.strip()]

columns_training = ['y'] + [f'feature_{i}' for i in range(1, len(training_data[0]))]
train_df = pd.DataFrame(training_data, columns=columns_training)

columns_testing = ['y'] + [f'feature_{i}' for i in range(1, len(testing_data[0]))]
test_df = pd.DataFrame(testing_data, columns=columns_testing)



model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='error',
    n_jobs = -1,
    device = 'cuda',
    random_state=42
)

X_train = train_df.drop(columns='y')
y_train = train_df['y']

model.fit(cd.DataFrame(X_train), cd.DataFrame(y_train))

exit()