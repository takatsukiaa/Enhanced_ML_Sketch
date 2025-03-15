#include "TopK.h"
#include "CMSketch.h"
#include <unordered_set>
// #include "CUSketch.h"
// #include "CUSACSketch.h"
// #include "PSketch.h"
// #include "PSACSketch.h"
#include <fstream>
#include <string>
//k is the size of heap (100 for now)
TopK topk(0,4,8192,500);
std::unordered_map<std::string, uint> actual_size;

//return actual top-k flows
std::vector<std::pair<std::string, uint>> GetActualTopK(uint k) {
    std::vector<std::pair<std::string, uint>> actual_list(actual_size.begin(), actual_size.end());
    std::sort(actual_list.begin(), actual_list.end(), 
        [](const std::pair<std::string, uint> &a, const std::pair<std::string, uint> &b) {
            return a.second > b.second; // 由大到小排序
        });
    if (actual_list.size() > k) {
        actual_list.resize(k); // 只取前 K 名
    }
    return actual_list;
}

//compare top-k flows
void CompareTopK(const std::vector<std::pair<std::string, uint>>& topk_estimated,
                 const std::vector<std::pair<std::string, uint>>& topk_actual) {
    std::unordered_set<std::string> estimated_set;
    std::unordered_set<std::string> actual_set;

    // 轉換為 Hash Set，方便比較
    for (const auto &pair : topk_estimated) {
        estimated_set.insert(pair.first);
    }
    for (const auto &pair : topk_actual) {
        actual_set.insert(pair.first);
    }

    // 計算 TP, FP, FN
    uint TP = 0, FP = 0, FN = 0;
    for (const auto &id : estimated_set) {
        if (actual_set.count(id)) {
            TP++; // 預測為 Top-K，且真的在 Top-K
        } else {
            FP++; // 預測為 Top-K，但實際不在 Top-K
        }
    }
    for (const auto &id : actual_set) {
        if (!estimated_set.count(id)) {
            FN++; // 真正的 Top-K 被漏掉了
        }
    }

    // Precision, Recall, F1-score
    float Precision = (TP + FP == 0) ? 0 : (float)TP / (TP + FP);
    float Recall = (TP + FN == 0) ? 0 : (float)TP / (TP + FN);
    float F1 = (Precision + Recall == 0) ? 0 : 2 * (Precision * Recall) / (Precision + Recall);

    // 輸出結果
	printf("TP: %u FP: %u\n", TP, FP);
	printf("FN: %u\n", FN);
    printf("Precision: %.4f\n", Precision);
    printf("Recall: %.4f\n", Recall);
    printf("F1-score: %.4f\n", F1);
}
int main(){
	std::string dat_path = "equinix-chicago1.dat";
	/*Insert your code here using the flowsize interface*/

	std::ifstream file(dat_path,std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	uint packet_count = 0;

	//將每個packet的data讀進來，並insert到sketch
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		topk.Insert(constData);
		actual_size[data]++;
		packet_count++;
	}
	file.close();


	file.open(dat_path,std::ios::binary);
	memset(buffer,0,13);
	// FILE* out = fopen("result.txt", "w");
	FILE* counters = fopen("equinix-chicago1_counters.txt","w");
	FILE* all_flows = fopen("equinix-chicago1_flows.txt","w");


	printf("Total Packet Count: %u\n", packet_count);
	printf("Total Flow Count: %lu\n", actual_size.size());

	//varify if top-k in heap is really top-k flow
	// 獲取 Min-Heap 的 Top-K
	std::vector<std::pair<std::string, uint>> estimated_topk = topk.GetTopK();

	// 獲取真實的 Top-K
	std::vector<std::pair<std::string, uint>> actual_topk = GetActualTopK(100); // 設定 K=100

	// 進行比較
	CompareTopK(estimated_topk, actual_topk);

	
	
	
	// fclose(out);
	fclose(counters);
	fclose(all_flows);
	return 0;
}