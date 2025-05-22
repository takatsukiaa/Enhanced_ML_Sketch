#include "CUSketch.h"
#include <fstream>
#include <string>
#include <queue>
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>
  
#define MICE_threshold 100
#define TEN_MINUTES 10000000 // base: 3982447
#define K 100

using namespace std;
struct Flow {
    string id = "";
    float size = 0;

    bool operator<(const Flow& other) const {
        return size > other.size; // 讓 priority_queue 變成最小堆
    }
};

static unordered_map<string, uint> actual_size;
static CUSketch* Sketch = new CUSketch(4, 8192);
set<Flow> actualSet;
set<Flow> predictSet;
int sock = -1;
static unordered_set<string> actual_id;

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

void Maintain_actualSet(string data, cuc* constData) {
    string flow_id = ToHex(data);
    int temp;
    Flow actual_temp = { flow_id, (float)actual_size[data] };
    // ------- Update actualSet -------
    // Check if a flow with the same id exists
    auto it_actual = find_if(actualSet.begin(), actualSet.end(), [&](const Flow& f) {
        return f.id == actual_temp.id;
    });
    if (it_actual != actualSet.end()){
        actualSet.erase(it_actual); // Remove old entry
    }
    
    actualSet.insert(actual_temp); // Insert new one
    actual_id.insert(actual_temp.id);

    if (actualSet.size() > K){
        string smallest_id = prev(actualSet.end())->id;
        actual_id.erase(smallest_id);
        actualSet.erase(prev(actualSet.end())); // Remove smallest
    }   

    
}

int main() {
    
    string dat_path = "equinix-chicago1.dat";
    int model_created=0;
    ifstream file(dat_path, ios::binary);
    if (!file) {
        cout << "Error, file not found!\n";
        return -1;
    }
    FILE *flows = fopen("/home/takatsukiaa/ML-Sketch/Python/TopK/training_flows.csv","w");
    unsigned char* buffer = new unsigned char[13];
    memset(buffer, 0, 13);
    
    uint packet_count = 0;
    uint num_of_flows = 0;
    int millions = 0;

    while (file.read(reinterpret_cast<char*>(buffer), 13) || file.gcount() > 0) {
        string data(reinterpret_cast<char*>(buffer), file.gcount());
        cuc* constData = buffer;
        if (packet_count < TEN_MINUTES) {
            Sketch->Insert(constData);
            actual_size[data]++;
            packet_count++;
            Maintain_actualSet(data, constData);
            continue;
        }
    }
}