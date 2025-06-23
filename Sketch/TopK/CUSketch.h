#ifndef CUSKETCH_H 
#define CUSKETCH_H
#include <climits>
#include "common.h"
#define THRESH 512
#define THRESH_BIT 9
#include <vector>
struct CUSketch{
public:
	CUSketch(uint d, uint w);
	~CUSketch();
	void Insert(cuc *str);
	void Enhanced_Insert(cuc* str);
	void GetHashedValue(cuc *str, uint *counters);
	uint Query(cuc *str, bool ml = FALSE);
	void PrintCounter(cuc* str, uint acc_val);
	void PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	float CalculateAAE(cuc * str, uint acc_val);
	void LoadPara(cuc *path = CMPATH);
	float Predict(uint *t);
	uint Enhanced_Query(cuc* str,int* feature_count);
	void Enhanced_PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	std::vector<int> GetCounter(cuc* str);
private:
	HashFunction *hf;
	uint** sketch;
	float* para;
	float* mean;
	float* scale;
	uint d;
	uint w;
	uint *t;
	uchar** ov_flags;
};
#endif