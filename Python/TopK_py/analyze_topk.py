import pandas as pd
import heapq

def analyze_topk(csv_file, topk_size=1000, output_file="topk_flows.csv"):
    """
    讀取 prediction 結果，根據 predicted_size 排出 Top-K flows。

    Args:
    - csv_file (str): 預測結果檔案 (必須有 ID, topk_flag, predicted_size, true_size 欄位)
    - topk_size (int): 要取幾個 Top-K (default=1000)
    - output_file (str): Top-K flow寫出的csv檔名
    """

    # 讀進資料
    df = pd.read_csv(csv_file)

    # 使用 heapq 選 TopK 最大的 predicted_size
    topk = heapq.nlargest(topk_size, df.to_dict(orient="records"), key=lambda x: x["predicted_size"])

    # 轉回DataFrame
    topk_df = pd.DataFrame(topk)

    # 存成新的CSV(optional)
    topk_df.to_csv(output_file, index=False)

    print(f"Top-{topk_size} flows saved to {output_file}")

    # 額外統計一下 TopK裡預測對不對（可選）
    true_positive_topk = (topk_df["topk_flag"] == 1).sum()
    print(f"In predicted Top-{topk_size}, there are {true_positive_topk} true TopK flows.")
    print(f"Precision: {true_positive_topk / topk_size:.4f}")

if __name__ == "__main__":
    # 可以改要分析的檔案
    analyze_topk(
        csv_file="training_prediction_result.csv",  # 預測結果檔
        topk_size=1000,                            # 要排Top多少（比如1000個flow）
        output_file="predicted_topk_flows.csv"      # 輸出TopK flow的檔案
    )