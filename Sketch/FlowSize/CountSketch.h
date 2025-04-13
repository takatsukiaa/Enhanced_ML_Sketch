#ifndef COUNTSKETCH_H
#define COUNTSKETCH_H
#include <climits>
#include <string_view>
#include <functional>
#include "common.h"
#define THRESH 28
struct CountSketch:public Sketch{
public:
	CountSketch(uint d, uint w);
	~CountSketch();
	void Insert(cuc *str);
	void Enhanced_Insert(cuc* str);
	void GetHashedValue(cuc *str, uint *counters);
    int Sign_Hash(cuc* str, int i);
	uint Query(cuc *str, bool ml = FALSE);
	void PrintCounter(cuc* str, uint acc_val);
	void PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
	float CalculateAAE(cuc * str, uint acc_val);
	void LoadPara(cuc *path = CMPATH);
	float Predict(uint *t);
	float CalculateAAE_ML(cuc * str, uint acc_val, float query_val);
	float CalculateARE(cuc * str, uint acc_val);
	uint Enhanced_Query(cuc* str,int* feature_count);
	void Enhanced_PrintCounterFile(cuc * str, uint acc_val, FILE * fout);
private:
	HashFunction *hf;
	int** sketch;
	float* para;
	float* mean;
	float* scale;
	uint d;
	uint w;
	uint *t;
	uchar** ov_flags;
};


CountSketch::CountSketch(uint d, uint w):d(d), w(w){
	sketch = new int*[d];
	ov_flags = new uchar*[d];
	for(uint i = 0; i < d; ++i){
		sketch[i] = new int[w];
		ov_flags[i] = new uchar[w];
		memset(sketch[i], 0, w * sizeof(int));
		memset(ov_flags[i], 0, w * sizeof(uchar));
	}
	hf = new HashFunction();
	para = new float[d];
	mean = new float[d];
	scale = new float[d];
	t = new uint[d];
}

CountSketch::~CountSketch(){
	for(uint i = 0; i < d; ++i) delete [] sketch[i];
	delete [] sketch;
	delete hf;
	delete [] para;
	delete [] mean;
	delete [] scale;
	delete [] t;
}

void CountSketch::Insert(cuc *str){
    for (uint i = 0; i < d; ++i){
        uint cid = hf->Str2Int(str, i) % w;
        if (sketch[i][cid] == -1) {
            return;
        }
		++sketch[i][cid];
	}
}
uint32_t fnv1a(cuc* key, size_t len, uint32_t seed) {
    uint32_t hash = 2166136261u ^ seed;  // offset basis XORed with seed
    for (size_t i = 0; i < len; ++i) {
        hash ^= key[i];
        hash *= 16777619u;  // FNV prime
    }
    return hash;
}

int CountSketch::Sign_Hash(cuc* str, int i)
{
    uint32_t h = fnv1a(str, 13, i + 100);
    return (h % 2 == 0) ? 1 : -1;
}

void CountSketch::Enhanced_Insert(cuc* str)
{
	uint min = 0;
	uint cid[4];
	for(uint i=0; i < d; i++)
	{
		cid[i] = hf->Str2Int(str, i) % w;

		if(sketch[i][cid[i]] == -1)
		{
			return;
		}
	}
	min = sketch[0][cid[0]];
	for(uint i = 0; i < d; i++)
	{
        int sign = Sign_Hash(str,i);
		if(ov_flags[i][cid[i]] == 0)
		{
			sketch[i][cid[i]] = sketch[i][cid[i]]<<22>>22;
		}
		sketch[i][cid[i]] += sign;
		if(sketch[i][cid[i]] < min || sketch[i][cid[i]] == min)
		{
			min = sketch[i][cid[i]];
		}
		if(sketch[i][cid[i]]>1023 && ov_flags[i][cid[i]] == 0)
		{
			ov_flags[i][cid[i]] = 1;
		}
	}
	for(uint i = 0; i < d; i++)
	{
		if(ov_flags[i][cid[i]] == '\0')
		{
			sketch[i][cid[i]] += min<<10;
		}
	}
}


void CountSketch::GetHashedValue(cuc *str, uint *counters)
{
	for(uint i = 0; i < d; i++)
	{
		counters[i] = hf->Str2Int(str,i) % w;
	}
}

uint CountSketch::Query(cuc *str, bool ml){
	memset(t, 0, sizeof(t)*3);

	uint Min;
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t,t+d);
	if (!ml || !need_analyze(t, d)) {
		Min = t[0];
		return Min;
	}
	else {
		printf("using ml:\n");
		std::sort(t, t+d);
		uint result = Predict(t);
		//if you want a float;
		if(t[3] - t[1] <= THRESH && result>0 && result <= t[0])
		{
			return result;
		}
		return t[0];
		// if(result > t[0] || result <= 0)
		// {
		// 	return t[0];
		// }
		// else
		// {
		// 	return result;
		// }
		//if you want a integer;
		//return (uint)Predict(t);
	}
}


void CountSketch::PrintCounter(cuc* str, uint acc_val){
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	if (need_analyze(t, d)) {
		std::sort(t, t + d);
		printf("Actual Size: ");
		printf("%u", acc_val);
		printf("Counter Values:");
		for(uint i = 0; i < d; ++i){
			printf(" %u", t[i]);
		}
		printf("\n");
	}
}
uint CountSketch::Enhanced_Query(cuc* str, int* feature_count)
{
    uint min = UINT_MAX;
    uint cid[3];
    
    // Step 1: 計算hash值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }

    // Step 2: 遍歷每一行，獲取計數器值
    for (uint i = 0; i < d; i++) {
        uint value;
        
        // 檢查溢出標誌
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，取整個 32 位值
            value = sketch[i][cid[i]];
			*feature_count +=1;
        } else {
            // 如果未溢出，僅取後 10 位值
            value = sketch[i][cid[i]] & 1023;
			*feature_count+=2;
        }
        
        // 更新最小值
        if (value < min) {
            min = value;
        }
    }

    // Step 3: 返回最小值
    return min;
}
void CountSketch::Enhanced_PrintCounterFile(cuc* str, uint acc_val, FILE* fout) {
    uint cid[4];
    int value;
    int feature_count = 0;
    //計算 Hash 值
    for (uint i = 0; i < d; i++) {
        cid[i] = hf->Str2Int(str, i) % w;
    }
	// 計算特徵數
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
           feature_count += 1;
        } else {
           feature_count += 2;
        }
    }
	fprintf(fout, "%d", feature_count);
    //寫入檔案
    fprintf(fout, " %u", acc_val); // 實際值
    for (uint i = 0; i < d; i++) {
        if (ov_flags[i][cid[i]] == 1) {
            // 如果溢出，輸出 32 位值
            value = sketch[i][cid[i]];
            fprintf(fout, " %d", value);
        } else {
            // 如果未溢出，輸出22位值 and 10-bit value
            int low = sketch[i][cid[i]] & 1023;
            int high = sketch[i][cid[i]] >> 10;
            fprintf(fout, " %d %d", high, low);
        }
    }
    fprintf(fout, "\n"); 
}


void CountSketch::PrintCounterFile(cuc * str, uint acc_val, FILE * fout)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	// fprintf(fout, "Actual Size: ");
	fprintf(fout, "%u ", acc_val);
	// fprintf(fout, "Counter Values:");
	for(uint i = 0; i < d-1; ++i){
		fprintf(fout, "%u ", t[i]);
	}
	fprintf(fout,"%u",t[d-1]);
	fprintf(fout, "\n");
}

void CountSketch::LoadPara(cuc *path){
	float temp;
	FILE *file = fopen((const char*)path, "r");
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", mean+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", scale+i);
	}
	fscanf(file, "%f", &temp);
	for(uint i = 0; i < d; ++i){
		fscanf(file, "%f", para+i);
	}
}
float CountSketch::CalculateAAE(cuc * str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs((float)t[0] - acc_val);
}
float CountSketch::CalculateAAE_ML(cuc * str, uint acc_val, float query_val)
{
	return abs(query_val - acc_val);
}
float CountSketch::CalculateARE(cuc* str, uint acc_val)
{
	for(uint i = 0; i < d; ++i){
		uint cid = hf->Str2Int(str, i)%w;
		t[i] = sketch[i][cid];
	}
	std::sort(t, t + d);
	return abs(((float)t[0] - acc_val)/acc_val);
}
float CountSketch::Predict(uint *t){
	float res = 0;
	for(uint i = 0; i < d; ++i){
		res += para[i]*((t[i]-mean[i])/scale[i]);
	}
	return res;
}

#endif