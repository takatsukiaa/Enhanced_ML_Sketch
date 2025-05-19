//this is the version of Basic TopK

#include "TopK.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <algorithm>
using namespace std;
std::unordered_map<std::string, uint> train_actual_size;
std::unordered_map<std::string, uint> test_actual_size;
int train_pkt_count = 10000000;
int initial_k = 250,max_k = 250,step = 100;
// 初始化 CSV 檔案
std::ofstream csv_file("topk_accuracy.csv");

void InitCSV() {
    csv_file << "File,Total Flow Count,Total Packet Count,k,Precision,Recall,F1-score\n";
}

// 記錄結果到 CSV
void SaveToCSV(const std::string& file_name, uint flow_count, uint packet_count, uint k, float precision, float recall, float f1) {
    csv_file << file_name << "," << flow_count << "," << packet_count << "," << k << "," << precision << "," << recall << "," << f1 << "\n";
}

// return actual top-k flows
std::vector<std::pair<std::string, uint>> GetActualTopK(uint k, std::unordered_map<std::string, uint> actual_size) {
    std::vector<std::pair<std::string, uint>> actual_list(actual_size.begin(), actual_size.end());
    std::sort(actual_list.begin(), actual_list.end(), 
        [](const std::pair<std::string, uint>& a, const std::pair<std::string, uint>& b) {
            return a.second > b.second;
        });
    if (actual_list.size() > k) {
        actual_list.resize(k);  // 只取前 K 名
    }
    return actual_list;
}

// 計算準確度
std::tuple<float, float, float> CompareTopK(const std::vector<std::pair<std::string, uint>>& topk_estimated,
                                            const std::vector<std::pair<std::string, uint>>& topk_actual) {
    std::unordered_set<std::string> estimated_set, actual_set;

    for (const auto& pair : topk_estimated) estimated_set.insert(pair.first);
    for (const auto& pair : topk_actual) actual_set.insert(pair.first);

    uint TP = 0, FP = 0, FN = 0;
    for (const auto& id : estimated_set) {
        if (actual_set.count(id)) TP++;
        else FP++;
    }
    for (const auto& id : actual_set) {
        if (!estimated_set.count(id)) FN++;
    }

    float precision = (TP + FP == 0) ? 0 : (float)TP / (TP + FP);
    float recall = (TP + FN == 0) ? 0 : (float)TP / (TP + FN);
    float f1 = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);
    int overlap_count = 0;
    for (const auto& pair : topk_estimated) {
        if (actual_set.count(pair.first)) overlap_count++;
    }
    float overlap_ratio = (float)overlap_count / topk_estimated.size();
    std::cout << "Overlap Ratio: " << overlap_ratio << std::endl;   
    return {precision, recall, f1};
}

int main() {
    std::string dat_path = "equinix-chicago1.dat";
    std::ifstream file(dat_path, std::ios::binary);
    if (!file) {
        std::cout << "Error: File not found!\n";
        return -1;
    }

    unsigned char* buffer = new unsigned char[13];
    memset(buffer, 0, 13);
    uint packet_count = 0;

    // 讀取數據一次，統計真實流量數據
    while (file.read(reinterpret_cast<char*>(buffer), 13) || file.gcount() > 0) {
        std::string data(reinterpret_cast<char*>(buffer), 13);
        
        packet_count++;
        if(packet_count <= train_pkt_count) {
            train_actual_size[data]++;
        }
        else {
            test_actual_size[data]++;
        }
    }
    file.close();
    //print pkt info    
    printf("Total Packet Count: %u\n", packet_count);
    printf("Training Flow Count: %lu\n", train_actual_size.size());
    printf("Testing Flow Count: %lu\n", test_actual_size.size());

    InitCSV();  // 初始化 CSV 檔案(for no ML)

    for (uint k = initial_k; k <= max_k; k += step) {
        // 重新建立 TopK
        TopK train_topk(0, 4, 8192/2, k);
        TopK test_topk(0, 4, 8192/2, k);
        
        std::unordered_set<std::string> train_actual_set;
        std::unordered_set<std::string> test_actual_set;
    
        for (const auto& pair : train_actual_size) train_actual_set.insert(pair.first);
        for(const auto& pair : test_actual_size) test_actual_set.insert(pair.first);
        train_topk.SetEnv(fopen("training.csv", "w"), &train_actual_set);
        test_topk.SetEnv(fopen("testing.csv", "w"), &test_actual_set);
		cout<<"**building topk....**"<<endl;
        // 插入流量數據到 TopK
        // 重新讀取檔案，將數據插入 TopK
        file.open(dat_path, std::ios::binary);
        memset(buffer, 0, 13);
        int size = 0;
        while (file.read(reinterpret_cast<char*>(buffer), 13) || file.gcount() > 0) {
            std::string data(reinterpret_cast<char*>(buffer),13);
            cuc* constData = buffer;
            size++;
            if(size <= train_pkt_count) {
                train_topk.Insert(constData);
            }
            else {
                test_topk.Insert(constData);
            }
            

        }
        file.close();
        train_topk.DumpToCSV("training.csv");
        test_topk.DumpToCSV("testing.csv");
		cout<<"**finish insert topk**"<<endl;
        // // 取得 Top-K（no ML)
        // std::vector<std::pair<std::string, uint>> estimated_topk = topk.GetTopK();
        // std::vector<std::pair<std::string, uint>> actual_topk = GetActualTopK(k,train_actual_size);

        // // 計算準確度
        // auto [precision, recall, f1] = CompareTopK(estimated_topk, actual_topk);
        // printf("K=%u -> Precision: %.4f, Recall: %.4f, F1-score: %.4f\n", k, precision, recall, f1);

        // // 記錄到 CSV
        // SaveToCSV(dat_path, train_actual_size.size(), packet_count, k, precision, recall, f1);
    }

    csv_file.close();
    delete[] buffer;
    return 0;
}