#include "CMSketch.h"
#include <fstream>
#include <string>
#include <queue>
#include <unordered_set>

#define MICE_threshold 100

using namespace std;

// 把 binary 字串轉成 hex 字串（方便當 ID）
string ToHex(const string& binary) {
    static const char* hex_chars = "0123456789ABCDEF";
    string hex;
    hex.reserve(binary.size() * 2);
    for (unsigned char c : binary) {
        hex += hex_chars[c >> 4];
        hex += hex_chars[c & 0x0F];
    }
    return hex;
}

// Flow 結構：存 flow id 和流量大小
struct Flow {
    string id;
    uint size;

    bool operator<(const Flow& other) const {
        return size > other.size; // 讓 priority_queue 變成最小堆
    }
};

int main() {
    string dat_path = "equinix-chicago1.dat";
    CMSketch* TrainSketch = new CMSketch(4, 8192);
    CMSketch* TestSketch = new CMSketch(4, 8192);

    ifstream file(dat_path, ios::binary);
    if (!file) {
        cout << "Error, file not found!\n";
        return -1;
    }

    unsigned char* buffer = new unsigned char[13];
    memset(buffer, 0, 13);
    uint packet_count = 0;
    int K = 1000; // Top-K大小設定

    unordered_map<string, uint> train_actual_size;
    unordered_map<string, uint> test_actual_size;

    // --- 讀檔，根據前後1千萬筆，分別插入不同的Sketch ---
    while (file.read(reinterpret_cast<char*>(buffer), 13) || file.gcount() > 0) {
        string data(reinterpret_cast<char*>(buffer), file.gcount());
        cuc* constData = buffer;

        if (packet_count < 10000000) {
            TrainSketch->Enhanced_Insert(constData);
            train_actual_size[data]++;
        } else {
            TestSketch->Enhanced_Insert(constData);
            test_actual_size[data]++;
        }
        packet_count++;
    }
    file.close();

    // --- 建立 Training Top-K ---
    priority_queue<Flow> trainHeap;
    for (auto& it : train_actual_size) {
        if (trainHeap.size() < K) trainHeap.push({it.first, it.second});
        else if (it.second > trainHeap.top().size) {
            trainHeap.pop();
            trainHeap.push({it.first, it.second});
        }
    }
    unordered_set<string> trainTopKFlows;
    while (!trainHeap.empty()) {
        trainTopKFlows.insert(trainHeap.top().id);
        trainHeap.pop();
    }

    // --- 建立 Testing Top-K ---
    priority_queue<Flow> testHeap;
    for (auto& it : test_actual_size) {
        if (testHeap.size() < K) testHeap.push({it.first, it.second});
        else if (it.second > testHeap.top().size) {
            testHeap.pop();
            testHeap.push({it.first, it.second});
        }
    }
    unordered_set<string> testTopKFlows;
    while (!testHeap.empty()) {
        testTopKFlows.insert(testHeap.top().id);
        testHeap.pop();
    }

    // --- 輸出 Training flows 到 CSV ---
    FILE* train_flows = fopen("equinix-chicago1_training_flows.csv", "w");
    if (!train_flows) {
        cout << "Error, cannot open training output file!\n";
        return -1;
    }
    fprintf(train_flows, "topk_flag,ID,feature_count,actual_size,1,2,3,4,5,6,7,8\n");

    for (auto it = train_actual_size.begin(); it != train_actual_size.end(); ++it) {
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
        int size = it->second;
        string hex_id = ToHex(it->first);

        int is_topk = trainTopKFlows.count(it->first) ? 1 : 0;
        fprintf(train_flows, "%d,%s", is_topk, hex_id.c_str());
        TrainSketch->Enhanced_PrintCounterFile(temp, size, train_flows);
    }
    fclose(train_flows);

    // --- 輸出 Testing flows 到 CSV ---
    FILE* test_flows = fopen("equinix-chicago1_testing_flows.csv", "w");
    if (!test_flows) {
        cout << "Error, cannot open testing output file!\n";
        return -1;
    }
    fprintf(test_flows, "topk_flag,ID,feature_count,actual_size,1,2,3,4,5,6,7,8\n");

    for (auto it = test_actual_size.begin(); it != test_actual_size.end(); ++it) {
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
        int size = it->second;
        string hex_id = ToHex(it->first);

        int is_topk = testTopKFlows.count(it->first) ? 1 : 0;
        fprintf(test_flows, "%d,%s", is_topk, hex_id.c_str());
        TestSketch->Enhanced_PrintCounterFile(temp, size, test_flows);
    }
    fclose(test_flows);

    printf("Total Packet Count: %u\n", packet_count);
    printf("Training Flow Count: %lu\n", train_actual_size.size());
    printf("Testing Flow Count: %lu\n", test_actual_size.size());

    delete[] buffer;
    delete TrainSketch;
    delete TestSketch;

    return 0;
}