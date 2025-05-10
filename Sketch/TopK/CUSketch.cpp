#include "CUSketch.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
using namespace std;
CUSketch::CUSketch(uint d, uint w):d(d), w(w){
	sketch = new uint*[d];
	ov_flags = new uchar*[d];
	for(uint i = 0; i < d; ++i){
		sketch[i] = new uint[w];
		ov_flags[i] = new uchar[w];
		memset(sketch[i], 0, w * sizeof(uint));
		memset(ov_flags[i], 0, w * sizeof(uchar));
	}
	hf = new HashFunction();
	para = new float[d];
	mean = new float[d];
	scale = new float[d];
	t = new uint[d];
}

CUSketch::~CUSketch(){
	for(uint i = 0; i < d; ++i) {
		delete [] sketch[i];
		delete [] ov_flags[i];
	}
	delete [] sketch;
	delete [] ov_flags;
	delete hf;
	delete [] para;
	delete [] mean;
	delete [] scale;
	delete [] t;
}

void CUSketch::Insert(cuc *str){
	memset(t, 0, sizeof(t));
	uint Min = INF_SHORT;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
		Min = min(Min, t[i]);
	}
	if (Min == INF_SHORT)
		return;
	for(uint i = 0; i < d; ++i){
		if (t[i]==Min){
			uint cid = hf->Str2Int(str, i)%w;
			++sketch[i][cid];
		}
	}
}

void CUSketch::Enhanced_Insert(cuc* str){
	uint min = 0;
	uint cid[4];
	uint value[4];
	for(uint i=0; i < d; i++){
		cid[i] = hf->Str2Int(str, i) % w;
		if(sketch[i][cid[i]] == -1) return;
	}
	min = sketch[0][cid[0]];
	for(uint i = 0; i < d; i++){
		if(ov_flags[i][cid[i]] == 0){
			sketch[i][cid[i]] = sketch[i][cid[i]]<<(32-THRESH_BIT)>>(32-THRESH_BIT);
		}
		if(sketch[i][cid[i]] <= min){
			min = sketch[i][cid[i]];
		}
	}
	uint min_after = 0;
	for(uint i = 0; i<d; i++){
		if(sketch[i][cid[i]] == min){
			sketch[i][cid[i]]++;
			min_after = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]] > THRESH-1 && ov_flags[i][cid[i]] == 0){
			ov_flags[i][cid[i]] = 1;
		}
	}
	for(uint i = 0; i < d; i++){
		if(ov_flags[i][cid[i]] == '\0'){
			sketch[i][cid[i]] += min_after << THRESH_BIT;
		}
	}
}

void CUSketch::Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout){
	uint cid[4];
	uint value;
	int feature_count = 0;
	for (uint i = 0; i < d; i++) cid[i] = hf->Str2Int(str, i) % w;
	for (uint i = 0; i < d; i++) feature_count += (ov_flags[i][cid[i]] == 1) ? 1 : 2;
	fprintf(fout, "%d", feature_count);
	fprintf(fout, " %u", acc_val);
	for (uint i = 0; i < d; i++) {
		if (ov_flags[i][cid[i]] == 1) {
			value = sketch[i][cid[i]];
			fprintf(fout, " %u", value);
		} else {
			uint low = sketch[i][cid[i]] & 511;
			uint high = sketch[i][cid[i]] >> THRESH_BIT;
			fprintf(fout, " %u %u", high, low);
		}
	}
	fprintf(fout, "\n");
}

void CUSketch::GetHashedValue(cuc *str, uint *counters){
	for(uint i = 0; i < d; i++){
		counters[i] = hf->Str2Int(str,i) % w;
	}
}

uint CUSketch::Query(cuc *str, bool ml){
	memset(t, 0, sizeof(t));
	uint Min = INF_SHORT;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
		Min = min(Min, t[i]);
	}
	if (!ml ) {
		return (uint)Min;
	} else {
		std::sort(t, t+d);
		float result = Predict(t);
		result = std::max((int)result, 1);
		result = result > t[0] ? t[0] : result;
		return result;
	}
}

uint CUSketch::Enhanced_Query(cuc* str, int* feature_count){
	uint min = UINT_MAX;
	uint cid[3];
	for (uint i = 0; i < d; i++) {
		cid[i] = hf->Str2Int(str, i) % w;
	}
	for (uint i = 0; i < d; i++) {
		uint value;
		if (ov_flags[i][cid[i]] == 1) {
			value = sketch[i][cid[i]];
			*feature_count += 1;
		} else {
			value = sketch[i][cid[i]] & 1023;
			*feature_count += 2;
		}
		if (value < min) {
			min = value;
		}
	}
	return min;
}

void CUSketch::PrintCounter(cuc* str, uint acc_val){
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	if (1) {
		std::sort(t, t+d);
		printf("%u", acc_val);
		for(uint i = 0; i < d; ++i){
			printf(" %u", t[i]);
		}
		printf("\n");
	}
}
vector<int> CUSketch::GetCounter(cuc* str){
	vector<int> result;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		result.push_back(sketch[i][cid]);
	}
	return result;
}
void CUSketch::PrintCounterFile(cuc * str, uint acc_val, FILE * fout){
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	fprintf(fout, "%u ", acc_val);
	for(uint i = 0; i < d-1; ++i){
		fprintf(fout, "%u ", t[i]);
	}
	fprintf(fout,"%u\n",t[d-1]);
}

float CUSketch::CalculateAAE(cuc * str, uint acc_val){
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return std::abs((float)t[0] - acc_val);
}

void CUSketch::LoadPara(cuc *path){
	FILE *file = fopen((const char*)path, "r");
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", mean+i);
	}
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", scale+i);
	}
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", para+i);
	}
}

float CUSketch::Predict(uint *t){
	float res = 0;
	for(uint i = 0; i < d; ++i){
		res += para[i]*(t[i]-mean[i])/scale[i];
	}
	return res;
}