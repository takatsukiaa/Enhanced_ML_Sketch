#include "CUSketch.h"
#include <fstream>
#include <string>
#include <queue>
#include <xgboost/c_api.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
  
#define MICE_threshold 100
#define TEN_MINUTES 7964894
#define K 100
#define safe_xgboost(call) \
  do { \
    int err = (call); \
    if (err != 0) { \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                          ": error in " + #call + ": " + XGBGetLastError()); \
    } \
  } while (0)
  

using namespace std;
struct Flow {
    string id = "";
    uint size = 0;

    bool operator<(const Flow& other) const {
        return size > other.size; // 讓 priority_queue 變成最小堆
    }
};

static unordered_map<string, uint> actual_size;
static CUSketch* TrainSketch = new CUSketch(4, 8192);
static CUSketch* TestSketch = new CUSketch(4, 8192);
set<Flow> actualSet;
set<Flow> predictSet;
static BoosterHandle booster4,booster5,booster6,booster7,booster8;
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

void Maintain_Set(string data, cuc* constData, int model_created, uint prediction) {
    string flow_id = ToHex(data);
    int temp;
    Flow actual_temp = { flow_id, actual_size[data] };
    Flow predict_temp = { flow_id, (model_created == 0) ? TestSketch->Enhanced_Query(constData, &temp) : prediction };

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

    // ------- Update predictSet -------
    auto it_predict = find_if(predictSet.begin(), predictSet.end(), [&](const Flow& f) {
        return f.id == predict_temp.id;
    });
    if (it_predict != predictSet.end())
        predictSet.erase(it_predict);

    predictSet.insert(predict_temp);
    if (predictSet.size() > K)
        predictSet.erase(prev(predictSet.end()));
}

int socket_initiation(){
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation error");
        return 1;
    }

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(50007);
    
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        perror("Invalid address");
        return 1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        return 1;
    }
    return 0;
}

float predict(vector<float> features){

    bst_ulong out_len;
    const float* out_result;
    DMatrixHandle dmat;
    safe_xgboost(XGDMatrixCreateFromMat(
        features.data(),          // pointer to data
        1,                        // 1 row
        features.size(),          // number of columns
        -1.0f,                    // missing value placeholder
        &dmat));                 // output handle
    switch(features.size()){
        case 4:
            safe_xgboost(XGBoosterPredict(booster4, dmat, 0, 0, 0,&out_len, &out_result));
            break;
        case 5:
            safe_xgboost(XGBoosterPredict(booster5, dmat, 0, 0, 0,&out_len, &out_result));
            break;
        case 6:
            safe_xgboost(XGBoosterPredict(booster6, dmat, 0, 0, 0,&out_len, &out_result));
            break;
        case 7:
            safe_xgboost(XGBoosterPredict(booster7, dmat, 0, 0, 0,&out_len, &out_result));
            break;
        case 8:
            safe_xgboost(XGBoosterPredict(booster8, dmat, 0, 0, 0,&out_len, &out_result));
            break;
        default:
            return -1;
            break;
    }
    safe_xgboost(XGDMatrixFree(dmat));
    return expm1(out_result[0]);
}

bool ExistsInBothSets(const string& id) {
    auto it = find_if(actualSet.begin(), actualSet.end(), [&](const Flow& f) {
        return f.id == id;
    });
    return it != actualSet.end();
}

void WriteSetToFile(const set<Flow>& flowSet, const string& label, ofstream& out) {
    out << "Top-K Flows in " << label << ":\n";
    for (const auto& f : flowSet)
        out << "ID: " << f.id << ", Size: " << f.size << '\n';
    out << '\n';
}

int main() {


    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster4));
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster5));
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster6));
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster7));
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster8));

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
    socket_initiation();

    while (file.read(reinterpret_cast<char*>(buffer), 13) || file.gcount() > 0) {
        string data(reinterpret_cast<char*>(buffer), file.gcount());
        cuc* constData = buffer;
        if (packet_count < TEN_MINUTES) {
            TrainSketch->Enhanced_Insert(constData);
            TestSketch->Enhanced_Insert(constData);
            actual_size[data]++;
            packet_count++;
            Maintain_Set(data, constData, model_created, 0);
            continue;
        }
        if(!model_created) {
            printf("10 Mins inserted, training model...\n");
            packet_count = 0;
            for (auto it = actual_size.begin(); it != actual_size.end(); ++it) {
                cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
                int second = it->second;
                TrainSketch->Enhanced_PrintCounterFile(temp, second, flows);
                num_of_flows++;
            }
            printf("First 10 minutes contained %u flows\n", num_of_flows);
            int flag = 1;
            send(sock, &flag, sizeof(flag), 0);
            int val_read = recv(sock, &model_created, sizeof(model_created), 0);
            if(val_read < 0) {
                perror("Failed to Receive");
            }
            if(model_created == -1){
                perror("Training Failed!");
            }
            if(close(sock) < 0){
                perror("Failed to close socket!");
            }
            // model_created = 1;
            printf("Training Completed! Loading Model...\n");
            safe_xgboost(XGBoosterLoadModel(booster4, "model_4.json"));
            safe_xgboost(XGBoosterLoadModel(booster5, "model_5.json"));
            safe_xgboost(XGBoosterLoadModel(booster6, "model_6.json"));
            safe_xgboost(XGBoosterLoadModel(booster7, "model_7.json"));
            safe_xgboost(XGBoosterLoadModel(booster8, "model_8.json"));

        }
        else {
            TestSketch->Enhanced_Insert(constData);
            actual_size[data]++;
            packet_count++;
            vector<float> hashed_value;
            hashed_value = TestSketch->GetHashedValue(constData);
            std::transform(hashed_value.begin(), hashed_value.end(), hashed_value.begin(), [](float x) {
                return std::log1p(x);  // float version of log(1 + x)
            });
            float prediction = predict(hashed_value);
            // printf("Actual Value:%u, Predicted Value:%f\n",actual_size[data], prediction);
            Maintain_Set(data, constData, model_created, prediction);
        }
    }
    printf("Analysis Completed\n");
    
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
        if (actual_id.count(pred.id)) {
            cout << "ID " << pred.id << " exists in both sets.\n";
            correct_count++;
        }
    }
    printf("The accuracy of the predict heap is: %d%% \n", correct_count);

    delete TrainSketch;
    delete TestSketch;
    return 0;
}