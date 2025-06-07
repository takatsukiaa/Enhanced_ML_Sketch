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
int initial_k = 100,max_k = 100,step = 100;

// CUSketch* train_sketch;
// CUSketch* test_sketch;

void ExportSketchFeaturesWithTopK(const std::string& output_file, uint k, CUSketch* sketch,
                                  const std::unordered_map<std::string, uint>& actual_size) {
    // 1. 先取得 top-k 真實流量 ID
    std::vector<std::pair<std::string, uint>> sorted_flows(actual_size.begin(), actual_size.end());
    std::sort(sorted_flows.begin(), sorted_flows.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    std::unordered_set<std::string> topk_ids;
    for (uint i = 0; i < k && i < sorted_flows.size(); ++i) {
        topk_ids.insert(sorted_flows[i].first);
    }

    // 2. 開啟輸出檔案
    FILE* fout = fopen(output_file.c_str(), "w");
    if (!fout) {
        std::cerr << "Failed to open file for writing: " << output_file << std::endl;
        return;
    }

    // 3. 寫入 header
    fprintf(fout, "flow_id,c1,c2,c3,c4,label\n");

    // 4. 遍歷所有 flow，查詢 counter 並輸出
    for (const auto& pair : actual_size) {
        const std::string& flow_id = pair.first;
        cuc* key = (cuc*)flow_id.c_str();
        std::vector<int> counters = sketch->GetCounter(key);  // 假設返回順序為 c1~c4

        int label = topk_ids.count(flow_id) ? 1 : 0;

        // 建議轉為 hex 字串更安全
        std::string hex_id;
        static const char* hex_chars = "0123456789ABCDEF";
        for (unsigned char c : flow_id) {
            hex_id += hex_chars[(c >> 4) & 0xF];
            hex_id += hex_chars[c & 0xF];
        }

        // 輸出一列資料
        fprintf(fout, "%s", hex_id.c_str());
        for (int c : counters) {
            fprintf(fout, ",%d", c);
        }
        fprintf(fout, ",%d\n", label);
    }

    fclose(fout);
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
        
        if(packet_count <= train_pkt_count) {
            train_actual_size[data]++;
        }
        else {
            test_actual_size[data]++;
        }
        packet_count++;
    }
    file.close();
    //print pkt info    
    printf("Total Packet Count: %u\n", packet_count);
    printf("Training Flow Count: %lu\n", train_actual_size.size());
    printf("Testing Flow Count: %lu\n", test_actual_size.size());


    for (uint k = initial_k; k <= max_k; k += step) {
        // 重新建立 TopK
        CUSketch* train_sketch = new CUSketch(4, 8192);
        CUSketch* test_sketch = new CUSketch(4, 8192*1.5);
        
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
                train_sketch->Insert(constData);
            }
            else {
                test_sketch->Insert(constData);
            }
            

        }
        file.close();

        //寫到csv中
        ExportSketchFeaturesWithTopK("../../Python/TopK/training.csv", k, train_sketch, train_actual_size);
        ExportSketchFeaturesWithTopK("../../Python/TopK/testing.csv", k, test_sketch, test_actual_size);
		cout<<"**finish insert topk**"<<endl;
        
        // 釋放記憶體
        delete train_sketch;
        delete test_sketch;

    }

    delete[] buffer;
    return 0;
}