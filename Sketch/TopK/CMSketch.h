#ifndef CMSKETCH_H
#define CMSKETCH_H

#include <climits>
#include <string>
#include "common.h"

#define THRESH 28

using namespace std;

struct CMSketch : public Sketch {
public:
    CMSketch(uint d, uint w);
    ~CMSketch();
    
    void Insert(cuc* str);
    void Enhanced_Insert(cuc* str);
    void GetHashedValue(cuc* str, uint* counters);
    uint Query(cuc* str, bool ml = FALSE);
    uint Enhanced_Query(cuc* str, int* feature_count);
    
    void PrintCounterFile(cuc* str, uint acc_val, FILE* fout);
    void Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout);
    
    float CalculateAAE(cuc* str, uint acc_val);
    float CalculateAAE_ML(cuc* str, uint acc_val, float query_val);
    float CalculateARE(cuc* str, uint acc_val);
    float Predict(uint* t);
    void LoadPara(cuc* path = CMPATH);

private:
    HashFunction* hf;
    uint** sketch;
    float* para;
    float* mean;
    float* scale;
    uint d;
    uint w;
    uint* t;
    uchar** ov_flags;
};

#endif // CMSKETCH_H