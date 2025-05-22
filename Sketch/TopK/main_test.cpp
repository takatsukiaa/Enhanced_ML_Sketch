#include "CMSketch.h"
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
#define K 250

using namespace std;
struct Flow {
    string id = "";
    uint size = 0;

    bool operator<(const Flow& other) const {
        return size > other.size; // 讓 priority_queue 變成最小堆
    }
};

static unordered_map<string, uint> actual_size;
static CMSketch* Sketch = new CMSketch(4, 8192);
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

void WriteSetToFile(const set<Flow>& flowSet, const string& label, ofstream& out) {
    out << "Top-K Flows in " << label << ":\n";
    for (const auto& f : flowSet)
        out << "ID: " << f.id << ", Size: " << f.size << '\n';
    out << '\n';
}

void Maintain_actualSet(string data) {
    string flow_id = ToHex(data);
    int temp;
    Flow actual_temp = { flow_id, actual_size[data] };
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
void Maintain_predictSet(string data, uint query_val) {
    
    string flow_id = ToHex(data);
    Flow temp = { flow_id,  query_val};
    auto it_actual = find_if(predictSet.begin(), predictSet.end(), [&](const Flow& f) {
        return f.id == temp.id;
    });
    if (it_actual != predictSet.end()){
        predictSet.erase(it_actual); // Remove old entry
    }
    predictSet.insert(temp);
    if (predictSet.size() > K){
        predictSet.erase(prev(predictSet.end())); // Remove smallest
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
        Sketch->Insert(constData);
        actual_size[data]++;
        packet_count++;
        if(actualSet.size() < K || actual_size[data] > prev(actualSet.end())->size)
            Maintain_actualSet(data);
        uint query_val = Sketch->Query(constData);
        if(predictSet.size()< K || query_val > prev(predictSet.end())->size){
            Maintain_predictSet(data, query_val);
        }
        memset(buffer, 0, sizeof(buffer));
    }
    printf("%lu\n",actual_id.size());
    ofstream outfile1("actual_heap.txt");
    if (!outfile1.is_open()) {
        cerr << "Failed to open output file.\n";
        return 1;
    }
    WriteSetToFile(actualSet,"actual heap", outfile1);
    outfile1.close();

    ofstream outfile2("predict_heap.txt");
    if (!outfile2.is_open()) {
        cerr << "Failed to open output file.\n";
        return 1;
    }
    WriteSetToFile(predictSet,"predict heap", outfile2);
    outfile2.close();

    int correct_count = 0;
    for (const Flow& pred : predictSet) {
        auto it_actual = find_if(actualSet.begin(), actualSet.end(), [&](const Flow& f) {
            return f.id == pred.id;
        });
        if (it_actual != actualSet.end()){
            correct_count++;
        }
    }
    printf("%d\n", correct_count);
    return 0;
}