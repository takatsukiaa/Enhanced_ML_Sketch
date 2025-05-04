import pandas as pd

# 讀進整個 CSV
def split_csv_by_feature_count(file_path, type):
    # 讀取 CSV 檔案
    df = pd.read_csv(file_path)

            
    # 按 feature_count 分成不同的 subset 並存檔
    for feature_count, output_file in {
        4: f"equinix-chicago1_output_4_{type}.csv",
        5: f"equinix-chicago1_output_5_{type}.csv",
        6: f"equinix-chicago1_output_6_{type}.csv",
        7: f"equinix-chicago1_output_7_{type}.csv",
        8: f"equinix-chicago1_output_8_{type}.csv"
    }.items():
        # 選出 feature_count == 指定數字的行
        subset = df[df["feature_count"] == feature_count]
        subset.to_csv(output_file, index=False)  # 寫出，不要加index欄

# 主程式
if __name__ == "__main__":
    # training
    split_csv_by_feature_count("equinix-chicago1_training_flows.csv",1)
    # testing
    split_csv_by_feature_count("equinix-chicago1_testing_flows.csv",2)