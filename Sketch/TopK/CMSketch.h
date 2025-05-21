#ifndef CMSKETCH_H
#define CMSKETCH_H

#include <vector>
#include "common.h"
#include <climits>
#define THRESH 28

struct CMSketch : public Sketch {
public:
	CMSketch(uint d, uint w);
	~CMSketch();

	void Insert(cuc *str);
	void Enhanced_Insert(cuc* str);
	void GetHashedValue(cuc *str, uint *counters);
	uint Query(cuc *str, bool ml = FALSE);
	// void PrintCounter(cuc* str, uint acc_val);
	void PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	float CalculateAAE(cuc * str, uint acc_val);
	void LoadPara(cuc *path = CMPATH);
	float Predict(uint *t);
	float CalculateAAE_ML(cuc * str, uint acc_val, float query_val);
	float CalculateARE(cuc * str, uint acc_val);
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